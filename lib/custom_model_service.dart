import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:flutter/foundation.dart'; // For debugPrint

class CustomModelService {
  tfl.Interpreter? _interpreter;
  List<String> _labels = [];
  List<int> _inputShape = [];
  List<int> _outputShape = [];

  Future<bool> ensureLoaded() async {
    if (_interpreter != null && _labels.isNotEmpty) return true;
    try {
      final modelData = await rootBundle.load('assets/model/vggish.tflite');
      _interpreter = tfl.Interpreter.fromBuffer(modelData.buffer.asUint8List(), options: tfl.InterpreterOptions()..threads = 2);

      _inputShape = _interpreter!.getInputTensor(0).shape;
      _outputShape = _interpreter!.getOutputTensor(0).shape;

      final labelsTxt = await rootBundle.loadString('assets/model/instrument_labels.txt');
      _labels = labelsTxt
          .split('\n')
          .where((l) => l.trim().isNotEmpty)
          .map((l) {
            final parts = l.trim().split(' ');
            if (parts.length >= 2) {
              // Teachable Machine format: "0 Arp" -> "Arp"
              return parts.sublist(1).join(' ');
            }
            return l.trim();
          })
          .toList();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<List<String>> analyzeFile(File wavFile) async {
    if (_interpreter == null) return [];

    final audio = await _readWavMonoFloat16k(wavFile);
    if (audio.isEmpty) return [];

    // Beklenen giriş: [1, frameLen] veya [frameLen]
    int frameLen;
    if (_inputShape.length == 2) {
      frameLen = _inputShape.last;
    } else if (_inputShape.length == 1) {
      frameLen = _inputShape.first;
    } else {
      // Desteklemediğimiz giriş şekli
      return [];
    }

    // Sessizlik kapısı (global)
    final rmsAll = _rms(audio, audio.length);
    final dbfsAll = 20 * math.log(rmsAll + 1e-12) / math.ln10;
    if (dbfsAll < -50) return [];

    final hop = math.max(1, (frameLen * 0.5).floor());
    final numClasses = _outputShape.last;
    final accum = List<double>.filled(numClasses, 0.0);
    int frames = 0;

    for (int start = 0; start < audio.length; start += hop) {
      final end = math.min(start + frameLen, audio.length);
      final take = end - start;
      if (take < frameLen ~/ 4) break;

      final frame = Float32List(frameLen);
      frame.setRange(0, take, audio.sublist(start, end));

      final rms = _rms(frame, take);
      final dbfs = 20 * math.log(rms + 1e-12) / math.ln10;
      if (dbfs < -55) continue;

      final input = (_inputShape.length == 2) ? frame.reshape([1, frameLen]) : frame;
      
      // Teachable Machine model çıktısını oku - basit format
      final output = Float32List(numClasses);
      final outputs = {0: output};
      
      _interpreter!.runForMultipleInputs([input], outputs);
      final List<double> scores = output.map((e) => e.toDouble()).toList();
      
      // Debug: İlk çerçeve skorlarını yazdır
      if (frames == 0) {
        debugPrint('First frame scores (first 10):');
        for (int i = 0; i < math.min(10, scores.length); i++) {
          debugPrint('  [$i]: ${scores[i]}');
        }
      }
      
      for (int i = 0; i < accum.length; i++) {
        accum[i] += scores[i];
      }
      frames++;
      if (frames >= 10) break;
    }

    if (frames == 0) return [];
    for (int i = 0; i < accum.length; i++) {
      accum[i] /= frames;
    }

    final indices = List<int>.generate(numClasses, (i) => i);
    indices.sort((a, b) => accum[b].compareTo(accum[a]));

    final best = indices.first;
    final bestScore = accum[best]; // Zaten normalize edilmiş
    final secondScore = accum[indices.length > 1 ? indices[1] : best];
    
    // Debug: Model çıktısını kontrol et
    debugPrint('Model output shape: ${_outputShape}');
    debugPrint('Number of classes: $numClasses');
    debugPrint('Labels length: ${_labels.length}');
    debugPrint('Accum length: ${accum.length}');
    debugPrint('Frames processed: $frames');
    
    // İlk 10 skoru yazdır
    debugPrint('First 10 raw scores:');
    for (int i = 0; i < math.min(10, accum.length); i++) {
      debugPrint('  [$i]: ${accum[i]}');
    }
    
    // Ilımlı eşikler: güven ve fark marjı
    if (bestScore < 0.35 || (bestScore - secondScore) < 0.10) return [];

    // Enstrüman sonuçlarından en fazla 5 tanesini döndür
    final top = <String>[];
    for (final i in indices) {
      if (i < _labels.length) top.add(_labels[i]);
      if (top.length == 5) break;
    }
    return top;
  }

  Future<Float32List> _readWavMonoFloat16k(File wav) async {
    final bytes = await wav.readAsBytes();
    if (bytes.length < 44) return Float32List(0);
    final data = ByteData.sublistView(bytes);
    final channels = data.getUint16(22, Endian.little);
    final sampleRate = data.getUint32(24, Endian.little);
    final bitsPerSample = data.getUint16(34, Endian.little);
    int offset = 12;
    int dataOffset = -1;
    int dataSize = 0;
    while (offset + 8 <= bytes.length) {
      final chunkId = String.fromCharCodes(bytes.sublist(offset, offset + 4));
      final chunkSize = data.getUint32(offset + 4, Endian.little);
      if (chunkId == 'data') { dataOffset = offset + 8; dataSize = chunkSize; break; }
      offset += 8 + chunkSize;
    }
    if (dataOffset < 0 || dataOffset + dataSize > bytes.length) return Float32List(0);
    final numSamples = (dataSize * 8) ~/ bitsPerSample;
    final mono = Float32List(numSamples ~/ channels);
    int write = 0;
    if (bitsPerSample == 16) {
      for (int i = 0; i < numSamples; i += channels) {
        double sum = 0.0;
        for (int c = 0; c < channels; c++) { sum += data.getInt16(dataOffset + ((i + c) * 2), Endian.little) / 32768.0; }
        mono[write++] = (sum / channels).clamp(-1.0, 1.0).toDouble();
      }
    } else if (bitsPerSample == 32) {
      for (int i = 0; i < numSamples; i += channels) {
        double sum = 0.0;
        for (int c = 0; c < channels; c++) { sum += data.getInt32(dataOffset + ((i + c) * 4), Endian.little) / 2147483648.0; }
        mono[write++] = (sum / channels).clamp(-1.0, 1.0).toDouble();
      }
    } else {
      return Float32List(0);
    }
    if (sampleRate == 16000) return mono.sublist(0, write);
    final targetLen = (write * 16000 / sampleRate).floor();
    final out = Float32List(targetLen);
    for (int i = 0; i < targetLen; i++) {
      final srcIndex = (i * sampleRate / 16000).floor();
      out[i] = mono[srcIndex.clamp(0, write - 1)];
    }
    return out;
  }

  double _rms(Float32List frame, int length) {
    double sum = 0.0; final n = math.max(1, length);
    for (int i = 0; i < n; i++) { final v = frame[i]; sum += v * v; }
    return math.sqrt(sum / n);
  }
}


