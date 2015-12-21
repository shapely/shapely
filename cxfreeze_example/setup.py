import sys
from cx_Freeze import setup, Executable

build_exe_options = {"packages": ["os", "shapely", "shapely.libgeos"], "excludes": ["tkinter"]}

setup(name="example", version="1.0",
      description="Example frozen shapely application",
      options={"build_exe": build_exe_options},
      executables=[
        Executable("example.py")
      ])
