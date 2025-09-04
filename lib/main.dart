import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:record/record.dart';
import 'dart:async';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'teachable_machine_service.dart';

void main() {
  runApp(const EnstrumanApp());
}

class EnstrumanApp extends StatelessWidget {
  const EnstrumanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Enstrüman Tespit',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      debugShowCheckedModeBanner: false,
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final AudioRecorder _recorder = AudioRecorder();
  bool _isRecording = false;
  String? _recordedPath;
  List<String> _detectedInstruments = [];
  int _selectedIndex = 0;
  final List<_HistoryItem> _history = [];
  TeachableMachineService? _custom;

  Future<void> _toggleRecord() async {
    if (_isRecording) {
      final path = await _recorder.stop();
      setState(() {
        _isRecording = false;
        _recordedPath = path;
      });
      await _analyzeRecording();
      return;
    }

    final hasPerm = await _recorder.hasPermission();
    if (!hasPerm) return;

    final dir = await getTemporaryDirectory();
    final filePath = Platform.isAndroid
        ? "${dir.path}/recording_${DateTime.now().millisecondsSinceEpoch}.m4a"
        : "${dir.path}/recording_${DateTime.now().millisecondsSinceEpoch}.m4a";

    await _recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: filePath,
    );
    setState(() {
      _isRecording = true;
      _recordedPath = null;
      _detectedInstruments = [];
    });
  }

  Future<void> _analyzeRecording() async {
    if (_recordedPath == null) return;
    try {
      _custom ??= TeachableMachineService();
      final ok = await _custom!.ensureLoaded();
      if (!ok) {
        setState(() {
          _detectedInstruments = ["Model bulunamadı"];
        });
      } else {
        final results = await _custom!.analyzeFile(File(_recordedPath!));
        setState(() {
          _detectedInstruments = results.isEmpty ? ["Bulunamadı"] : results;
        });
      }
    } catch (e) {
      setState(() {
        _detectedInstruments = ["Analiz hatası: $e"];
      });
    } finally {
      setState(() {
        _history.insert(
          0,
          _HistoryItem(
            dateTime: DateTime.now(),
            filePath: _recordedPath ?? "",
            instruments: List<String>.from(_detectedInstruments),
          ),
        );
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Enstrüman Tespit"),
      ),
      body: _selectedIndex == 0 ? _buildHomeBody(context) : _buildHistoryBody(context),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => setState(() => _selectedIndex = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.home_outlined), selectedIcon: Icon(Icons.home), label: 'Ev'),
          NavigationDestination(icon: Icon(Icons.history), selectedIcon: Icon(Icons.history_toggle_off), label: 'Geçmiş'),
        ],
      ),
    );
  }

  Widget _buildHomeBody(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          const SizedBox(height: 16),
          Expanded(
            child: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _MicAnimatedButton(
                    isRecording: _isRecording,
                    onTap: _toggleRecord,
                    color: scheme.primary,
                  ),
                  const SizedBox(height: 24),
                  Text(
                    _isRecording ? "Kaydediliyor..." : "Mikrofona basıp kaydı başlat",
                    style: Theme.of(context).textTheme.titleMedium,
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          ),
          if (_detectedInstruments.isNotEmpty)
            Card(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("Bulunan Enstrümanlar", style: Theme.of(context).textTheme.titleLarge),
                    const SizedBox(height: 8),
                    ..._detectedInstruments.map((e) => ListTile(
                          leading: const Icon(Icons.music_note),
                          title: Text(e),
                        )),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildHistoryBody(BuildContext context) {
    if (_history.isEmpty) {
      return Center(
        child: Text(
          'Henüz kayıt yok',
          style: Theme.of(context).textTheme.titleMedium,
        ),
      );
    }
    return ListView.separated(
      padding: const EdgeInsets.all(16),
      itemCount: _history.length,
      separatorBuilder: (_, __) => const SizedBox(height: 8),
      itemBuilder: (context, index) {
        final item = _history[index];
        return Card(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: ListTile(
            leading: const Icon(Icons.record_voice_over),
            title: Text(
              item.instruments.isEmpty ? 'Analiz bekleniyor' : item.instruments.join(', '),
            ),
            subtitle: Text(item.dateTime.toLocal().toString()),
            trailing: const Icon(Icons.chevron_right),
          ),
        );
      },
    );
  }
}

class _MicAnimatedButton extends StatefulWidget {
  final bool isRecording;
  final VoidCallback onTap;
  final Color color;
  const _MicAnimatedButton({required this.isRecording, required this.onTap, required this.color});

  @override
  State<_MicAnimatedButton> createState() => _MicAnimatedButtonState();
}

class _MicAnimatedButtonState extends State<_MicAnimatedButton>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _scale;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 900));
    _scale = Tween(begin: 1.0, end: 1.08).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));
  }

  @override
  void didUpdateWidget(covariant _MicAnimatedButton oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isRecording && !_controller.isAnimating) {
      _controller.repeat(reverse: true);
    } else if (!widget.isRecording && _controller.isAnimating) {
      _controller.stop();
      _controller.reset();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final primary = widget.color;
    return GestureDetector(
      onTap: widget.onTap,
      child: ScaleTransition(
        scale: _scale,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          width: 180,
          height: 180,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: widget.isRecording ? primary : primary.withOpacity(0.85),
            boxShadow: [
              BoxShadow(
                color: primary.withOpacity(widget.isRecording ? 0.5 : 0.25),
                blurRadius: widget.isRecording ? 40 : 16,
                spreadRadius: widget.isRecording ? 8 : 2,
              ),
            ],
          ),
          child: Icon(
            widget.isRecording ? Icons.stop : Icons.mic,
            size: 64,
            color: Theme.of(context).colorScheme.onPrimary,
          ),
        ),
      ),
    );
  }
}

class _HistoryItem {
  final DateTime dateTime;
  final String filePath;
  final List<String> instruments;
  _HistoryItem({required this.dateTime, required this.filePath, required this.instruments});
}
