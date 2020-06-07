# Introduction

This project aims to evaluate different methods of extraction and classification in order to develop the most efficient speech recognition system possible.

# Installation

## The Speaker In the Wild database

To use this project, you must have a copy of the database *Speaker in the Wild*. You can follow the instructions at this [link](http://www.speech.sri.com/projects/sitw/). Unzip the database in the folder `database`, located at the root of the project.

## Libraries

The project contains certain dependencies to python libraries and requires Python 3.6 or above.
You can install these librairies and this version of python in a specific virtual environment.
On Linux, this can be achieved by typing the commands on a Linux/Mac terminal on the root of the project

```bash
python3 -m venv
source venv/bin/activate
pip3 install -r requirements.txt
```

**Note** The Soundfile library cannot currently read flac audio file on Windows,
the project can still be used on Windows by downloading *Ubuntu* from the Windows store.

# Execution

You can execute the project by simply run the command : 
```bash
python3 main.py
```
