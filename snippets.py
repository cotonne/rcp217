########################################################
### Generate MIDI file                               ###
########################################################

from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(Message('program_change', program=12, time=0))
track.append(Message('note_on', note=69, velocity=27, time=12))
track.append(Message('note_off', note=69, velocity=64, time=4))

#{'type': 'note_on', 'time': 12, 'note': 69, 'velocity': 27, 'channel': 9}
#{'type': 'note_off', 'time': 4, 'note': 69, 'velocity': 64, 'channel': 9}


mid.save('new_song.mid')