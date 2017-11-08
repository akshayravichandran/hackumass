import os
from shutil import copyfile

for fold in list(os.walk('audio'))[1:]:
    dir = fold[0]
    files = fold[2]
    num_siren = 0
    num_other = 0
    print("Process folder " + dir)
    for file in files:
        if '.wav' in file:
            if file.split('-')[1] is '8':
                copyfile(dir + '\\' + file, 'audio_balanced\\' + dir.split('\\')[1] + '\\' + file)
                num_siren += 1
            elif num_other - num_siren <= 50:
                copyfile(dir + '\\' + file, 'audio_balanced\\' + dir.split('\\')[1] + '\\' + file)
                num_other += 1