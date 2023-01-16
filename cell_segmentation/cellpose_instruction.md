For MacOS M1, qt5 installation doesn't work through pip so this needs to be run first

```
brew install qt5
export PATH="/opt/homebrew/opt/qt5/bin:$PATH"
python3 -m ensurepip --default-pip
pip3 install pyqt5-sip
pip3 install pyqt5 --config-settings --confirm-license= --verbose
```

## Download and run GUI

```
conda create --name cellpose python=3.8
conda activate cellpose
pip3 install 'cellpose[gui]'
python -m cellpose
```

Drag and drop image in GUI and run model

## Download Cellpose

```
pip3 install cellpose
```

Supplementary:
Look at Colab notebook for code:
https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb

Youtube tutorial:
https://www.youtube.com/watch?v=5qANHWoubZU
