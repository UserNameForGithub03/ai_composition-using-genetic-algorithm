from music21 import *
from music21 import chord
import os

#设置你的MuseScore路径
env = environment.Environment()
environment.Environment()['musicxmlPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'
environment.Environment()['musescoreDirectPNGPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'

# 设置 MIDI 文件所在目录
midi_dir = 'D:\\2(2)\\音乐与数学\\data'

# 设置输出文本文件路径
output_file = 'D:\\2(2)\\音乐与数学\\output.txt'

# 打开输出文件，准备写入数据
with open(output_file, 'w') as f:
    # 遍历 MIDI 文件所在目录下的所有文件
    for file in os.listdir(midi_dir):
        # 检查文件是否是 MIDI 文件
        if file.endswith('.mid'):
            # 读取 MIDI 文件中的音符
            stream0 = converter.parse(os.path.join(midi_dir, file))
            notes_to_parse = None

            # 获取乐器部分
            parts = instrument.partitionByInstrument(stream0)

            if parts: # 如果有乐器部分，取第一个乐器
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = stream0.flat.notes # 没有乐器部分，纯音符组成
            
            wanted_notes = stream.Stream()  # Create a new stream for the wanted notes
            
            #筛选出音高以4、5、6结尾的音符
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    if element.pitch.nameWithOctave.endswith(('4','5','6')):
                        wanted_notes.append(element)
                    elif isinstance(element, chord.Chord):
                        pitches = element.sortAscending().pitches
                        if any(pitch.nameWithOctave.endswith(('4','5','6')) for pitch in pitches):
                            wanted_notes.append(element)
            
            #播放筛选后的音符序列如果播放不了的话大概可以在这种地方找到C:\Users\Lenovo\AppData\Local\Temp\music21
            # wanted_notes.show('midi')
            
            # 遍历音符，写入到输出文件中（在output.txt看看筛选后都有哪些音符）
            for element in wanted_notes:
                if isinstance(element, note.Note):
                    # 如果是 Note 类型，写入音调
                    f.write(str(element.pitch) + ' ')
                elif isinstance(element, chord.Chord):
                    # 如果是 Chord 类型，写入和弦
                    f.write(str(element.sortAscending().pitches[-1]) + ' ')
            
            # 写入分隔符
            f.write('\n\n')
