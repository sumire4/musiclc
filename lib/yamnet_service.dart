import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

class YamnetService {
  tfl.Interpreter? _interpreter;
  List<String> _labels = [];

  Future<bool> ensureLoaded() async {
    if (_interpreter != null && _labels.isNotEmpty) return true;
    try {
      final modelData = await rootBundle.load('assets/model/yamnet.tflite');
      final options = tfl.InterpreterOptions()
        ..threads = 2;
      _interpreter = tfl.Interpreter.fromBuffer(modelData.buffer.asUint8List(), options: options);
      final csv = await rootBundle.loadString('assets/model/yamnet_class_map.csv');
      _labels = csv
          .split('\n')
          .skip(1)
          .where((l) => l.trim().isNotEmpty)
          .map((l) => _toTurkishLabel(l.split(',').last.trim()))
          .toList();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<List<String>> analyzeFile(File wavFile) async {
    if (_interpreter == null) return [];
    final floatBuffer = await _readWavMonoFloat16k(wavFile);
    if (floatBuffer.isEmpty) return [];

    // YAMNet input: 15600 örnek (0.975s). Sliding window ile birden çok kareden skorları toplayıp ortala.
    const frameLen = 15600;
    const hop = 8000; // ~0.5 sn kayma
    final totalLen = floatBuffer.length;

    // Sessizlik kapısı: tüm sinyalin RMS’ini kontrol et
    final rmsAll = _rms(floatBuffer, totalLen);
    final dbfsAll = 20 * math.log(rmsAll + 1e-12) / math.ln10;
    if (dbfsAll < -50) return [];

    final accum = List<double>.filled(521, 0.0);
    int frames = 0;
    for (int start = 0; start + 1 < totalLen && frames < 10; start += hop) {
      final end = math.min(start + frameLen, totalLen);
      final take = end - start;
      if (take <= frameLen ~/ 4) break; // çok kısa pencereyi atla
      final frame = Float32List(frameLen);
      frame.setRange(0, take, floatBuffer.sublist(start, end));

      // Pencere RMS’ine göre lokal sessizlik kapısı
      final rms = _rms(frame, take);
      final dbfs = 20 * math.log(rms + 1e-12) / math.ln10;
      if (dbfs < -55) continue;

      final input = frame.reshape([1, frameLen]);
      final outputScores = List<double>.filled(1 * 521, 0.0).reshape([1, 521]);
      final outputEmbeddings = List<double>.filled(1 * 1024, 0.0).reshape([1, 1024]);
      final outputPatches = List<double>.filled(1 * 1, 0.0).reshape([1, 1]);
      final outputs = {0: outputScores, 1: outputEmbeddings, 2: outputPatches};
      _interpreter!.runForMultipleInputs([input], outputs);
      final List<double> sc = outputScores[0];
      for (int i = 0; i < accum.length; i++) {
        accum[i] += sc[i];
      }
      frames++;
    }

    if (frames == 0) return [];
    for (int i = 0; i < accum.length; i++) {
      accum[i] /= frames;
    }

    final indices = List<int>.generate(accum.length, (i) => i);
    indices.sort((a, b) => accum[b].compareTo(accum[a]));

    // Daha sıkı güven eşiği ve fark marjı
    final best = indices.first;
    final bestScore = accum[best];
    final secondScore = accum[indices[1]];
    if (bestScore < 0.35 || (bestScore - secondScore) < 0.10) {
      return [];
    }

    final topFiltered = <String>[];
    for (final i in indices) {
      if (_labels.isEmpty || i >= _labels.length) continue;
      final label = _labels[i];
      if (_isInstrumentLabel(label)) {
        topFiltered.add(label);
        if (topFiltered.length == 5) break;
      }
    }
    return topFiltered;
  }

  String _toTurkishLabel(String en) {
    final key = en.trim();
    const map = {
      'Guitar': 'Gitar',
      'Acoustic guitar': 'Akustik gitar',
      'Electric guitar': 'Elektro gitar',
      'Bass guitar': 'Bas gitar',
      'Piano': 'Piyano',
      'Organ': 'Org',
      'Synthesizer': 'Synthesizer',
      'Saxophone': 'Saksafon',
      'Trumpet': 'Trompet',
      'Trombone': 'Trombon',
      'French horn': 'Korno',
      'Tuba': 'Tuba',
      'Flute': 'Flüt',
      'Recorder': 'Blok flüt',
      'Clarinet': 'Klarnet',
      'Harp': 'Arp',
      'Violin, fiddle': 'Keman',
      'Violin': 'Keman',
      'Viola': 'Viyola',
      'Cello': 'Çello',
      'Double bass': 'Kontrbas',
      'Drum kit': 'Bateri',
      'Drum': 'Davul',
      'Snare drum': 'Trampet',
      'Bass drum': 'Bas davul',
      'Cymbal': 'Zil',
      'Hi-hat': 'Hi-hat',
      'Tambourine': 'Tef',
      'Maraca': 'Marakas',
      'Harmonica': 'Mızıka',
      'Accordion': 'Akordeon',
      'Voice': 'Vokal',
      'Female singing': 'Kadın vokal',
      'Male singing': 'Erkek vokal',
      'Choir': 'Koro',
      'Choir, vocal ensemble': 'Koro',
      'Singing': 'Vokal',
      'Rapping': 'Rap',
      'Beatboxing': 'Beatbox',
    };
    if (map.containsKey(key)) return map[key]!;
    final cleaned = key.replaceAll('_', ' ');
    return cleaned.isEmpty ? key : cleaned[0].toUpperCase() + cleaned.substring(1);
  }

  Future<Float32List> _readWavMonoFloat16k(File wav) async {
    final bytes = await wav.readAsBytes();
    if (bytes.length < 44) return Float32List(0);
    final data = ByteData.sublistView(bytes);
    // WAV header parsing (little-endian)
    final channels = data.getUint16(22, Endian.little);
    final sampleRate = data.getUint32(24, Endian.little);
    final bitsPerSample = data.getUint16(34, Endian.little);
    // Find 'data' chunk
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
    if (dataOffset < 0 || dataOffset + dataSize > bytes.length) return Float32List(0);

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

    // Resample to 16k if needed (very simple nearest-neighbor for speed)
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

  bool _isInstrumentLabel(String labelTr) {
    // Türkçeleştirilmiş etiketler üzerinden basit beyaz liste filtreleme
    const allow = {
      'Gitar', 'Akustik gitar', 'Elektro gitar', 'Bas gitar',
      'Piyano', 'Org', 'Synthesizer', 'Saksafon', 'Trompet', 'Trombon', 'Korno', 'Tuba',
      'Flüt', 'Blok flüt', 'Klarnet', 'Arp', 'Keman', 'Viyola', 'Çello', 'Kontrbas',
      'Bateri', 'Davul', 'Trampet', 'Bas davul', 'Zil', 'Hi-hat', 'Tef', 'Marakas', 'Mızıka', 'Akordeon'
    };
    return allow.contains(labelTr);
  }

  double _rms(Float32List frame, int length) {
    double sum = 0.0;
    final n = math.max(1, length);
    for (int i = 0; i < n; i++) {
      final v = frame[i];
      sum += v * v;
    }
    return math.sqrt(sum / n);
  }
}
