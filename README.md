# RCP217 - Génération de musiques

## Jeu de données

 - Récupération du [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)


    $ wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
 
 - Extraction d'un fichier MIDI


     $ tar -xvf lmd_matched.tar.gz lmd_matched/L/Z/U/TRLZURC128E079376E/cad555c70af4bd043445920c8bcb4b00.mid

 - Lecture d'un fichier MIDI sous Linux/Debian


     $ sudo apt install fluidsynth fluid-soundfont-gm
     $ fluidsynth -i -a pulseaudio lmd_matched/L/Z/U/TRLZURC128E079376E/cad555c70af4bd043445920c8bcb4b00.mid

## Projet

Le projet est prévu pour fonctionner avec python 3.8+

### Installation

    python3 -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt
