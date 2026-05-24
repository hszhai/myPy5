"""Hello World with py5 — standalone script.

Run from a Terminal (not inside VS Code's notebook):

    ~/miniconda3/bin/python hello_world.py

This opens a real sketch window. Close the window to stop the sketch.
"""
import os

# py5 requires Java 17. Point JAVA_HOME at the Homebrew openjdk@17 install
# so this script works no matter what JAVA_HOME the shell has.
os.environ["JAVA_HOME"] = "/usr/local/opt/openjdk@17"

import py5


def setup():
    py5.size(400, 300)


def draw():
    py5.background(255)
    py5.fill(0, 100, 255)
    py5.rect(50, 50, 100, 100)
    py5.fill(255, 100, 0)
    py5.circle(300, 150, 50)


py5.run_sketch()
