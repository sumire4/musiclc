import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:flutter/foundation.dart';

class TeachableMachineService {
  tfl.Interpreter? _interpreter;
  List<String> _labels = [];
  List<int> _inputShape = [];
  List<int> _outputShape = [];

  Future<bool> ensureLoaded() async {
    if (_interpreter != null && _labels.isNotEmpty) return true;
    
    try {
      // Model yükle
      final modelData = await rootBundle.load('assets/model/vggish.tflite');
      final options = tfl.InterpreterOptions()..threads = 2;
      _interpreter = tfl.Interpreter.fromBuffer(modelData.buffer.asUint8List(), options: options);

      // Input/Output şekillerini al
      _inputShape = _interpreter!.getInputTensor(0).shape;
      _outputShape = _interpreter!.getOutputTensor(0).shape;

      debugPrint('Input shape: $_inputShape');
      debugPrint('Output shape: $_outputShape');

      // Label dosyasını oku - Teachable Machine formatı
      final labelsTxt = await rootBundle.loadString('assets/model/instrument_labels.txt');
      _labels = labelsTxt
          .split('\n')
          .where((l) => l.trim().isNotEmpty)
          .map((l) {
            final parts = l.trim().split(' ');
            if (parts.length >= 2) {
              // "0 Arp" -> "Arp"
              return parts.sublist(1).join(' ');
            }
            return l.trim();
          })
          .toList();

      debugPrint('Labels loaded: $_labels');
      return true;
    } catch (e) {
      debugPrint('Error loading model: $e');
      return false;
    }
  }

  Future<List<String>> analyzeFile(File wavFile) async {
    if (_interpreter == null) return [];

    try {
      // Ses dosyasını oku
      final audio = await _readWavMonoFloat16k(wavFile);
      if (audio.isEmpty) return [];

      // Sessizlik kontrolü - eşiği düşür
      final rms = _calculateRMS(audio);
      debugPrint('Audio RMS: ${rms.toStringAsFixed(2)} dBFS');
      if (rms < -70.0) { // Çok daha düşük eşik
        debugPrint('Audio too quiet (RMS: ${rms.toStringAsFixed(2)} dBFS)');
        return [];
      }

      // Model için input hazırla
      final inputLength = _inputShape.last;
      final input = Float32List(inputLength);
      
      // Audio'yu input'a kopyala (padding veya truncation)
      final copyLength = math.min(audio.length, inputLength);
      input.setRange(0, copyLength, audio.sublist(0, copyLength));

      // Input'u reshape et
      final inputReshaped = input.reshape([1, inputLength]);

      // Output hazırla - tek Float32List olarak
      final outputLength = _outputShape.last;
      final output = Float32List(outputLength);

      // Model çalıştır - runForMultipleInputs kullan
      final outputs = {0: output};
      _interpreter!.runForMultipleInputs([inputReshaped], outputs);

      // Sonuçları işle
      final scores = output.map((e) => e.toDouble()).toList();
      
      debugPrint('Raw scores: $scores');

      // En yüksek skorları bul
      final indices = List<int>.generate(scores.length, (i) => i);
      indices.sort((a, b) => scores[b].compareTo(scores[a]));

      // En yüksek 3 skoru al - eşiği düşür
      final results = <String>[];
      for (int i = 0; i < math.min(3, indices.length); i++) {
        final idx = indices[i];
        final score = scores[idx];
        debugPrint('${_labels[idx]}: ${score.toStringAsFixed(4)}');
        if (idx < _labels.length && score > 0.01) { // Çok düşük eşik
          results.add(_labels[idx]);
        }
      }

      debugPrint('Results: $results');
      return results;
    } catch (e) {
      debugPrint('Error analyzing file: $e');
      return [];
    }
  }

  double _calculateRMS(Float32List buffer) {
    if (buffer.isEmpty) return -100.0;
    double sumOfSquares = 0.0;
    for (final sample in buffer) {
      sumOfSquares += sample * sample;
    }
    final rms = math.sqrt(sumOfSquares / buffer.length);
    return 20 * math.log(rms / 1.0) / math.ln10;
  }

  Future<Float32List> _readWavMonoFloat16k(File wav) async {
    final bytes = await wav.readAsBytes();
    if (bytes.length < 44) return Float32List(0);
    
    final data = ByteData.sublistView(bytes);
    final channels = data.getUint16(22, Endian.little);
    final sampleRate = data.getUint32(24, Endian.little);
    final bitsPerSample = data.getUint16(34, Endian.little);

    // Data chunk'ı bul
    int offset = 12;
    int dataOffset = -1;
    int dataSize = 0;
    
    while (offset + 8 <= bytes.length) {
      final chunkId = String.fromCharCodes(bytes.sublist(offset, offset + 4));
      final chunkSize = data.getUint32(offset + 4, Endian.little);
      if (chunkId == 'data') {
        dataOffset = offset + 8;
        dataSize = chunkSize;
        break;
      }
      offset += 8 + chunkSize;
    }

    if (dataOffset < 0 || dataOffset + dataSize > bytes.length) {
      return Float32List(0);
    }

    final numSamples = (dataSize * 8) ~/ bitsPerSample;
    final mono = Float32List(numSamples ~/ channels);

    int write = 0;
    if (bitsPerSample == 16) {
      for (int i = 0; i < numSamples; i += channels) {
        double sum = 0.0;
        for (int c = 0; c < channels; c++) {
          final sample = data.getInt16(dataOffset + ((i + c) * 2), Endian.little);
          sum += sample / 32768.0;
        }
        mono[write++] = (sum / channels).clamp(-1.0, 1.0).toDouble();
      }
    } else if (bitsPerSample == 32) {
      for (int i = 0; i < numSamples; i += channels) {
        double sum = 0.0;
        for (int c = 0; c < channels; c++) {
          final sample = data.getInt32(dataOffset + ((i + c) * 4), Endian.little);
          sum += sample / 2147483648.0;
        }
        mono[write++] = (sum / channels).clamp(-1.0, 1.0).toDouble();
      }
    } else {
      return Float32List(0);
    }

    // 16kHz'e resample et
    if (sampleRate == 16000) {
      return mono.sublist(0, write);
    }
    
    final targetLen = (write * 16000 / sampleRate).floor();
    final out = Float32List(targetLen);
    for (int i = 0; i < targetLen; i++) {
      final srcIndex = (i * sampleRate / 16000).floor();
      out[i] = mono[srcIndex.clamp(0, write - 1)];
    }
    
    return out;
  }
}
