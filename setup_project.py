# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 14:14:10 2025

@author: heba_
"""


import os
from pathlib import Path

def create_project_structure():
    """Create the complete project structure."""
    
    print("🚀 Setting up BiGKbhb project structure...")
    
    # Define the project structure
    directories = [
        "src/data",
        "src/models", 
        "src/utils",
        "scripts",
        "notebooks",
        "tests",
        "data/raw",
        "models/saved_models",
        "results"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py", 
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created file: {init_file}")
    
    # Create .gitkeep files for empty directories
    gitkeep_files = [
        "data/raw/.gitkeep",
        "models/saved_models/.gitkeep",
        "results/.gitkeep"
    ]
    
    for gitkeep_file in gitkeep_files:
        Path(gitkeep_file).touch()
        print(f"✅ Created: {gitkeep_file}")
    
    print("\n🎉 Project structure created successfully!")
    print("\nCurrent structure:")
    print("BiGKbhb/")
    print("├── src/")
    print("│   ├── data/")
    print("│   ├── models/")
    print("│   └── utils/")
    print("├── scripts/")
    print("├── notebooks/")
    print("├── tests/")
    print("├── data/raw/")
    print("├── models/saved_models/")
    print("└── results/")

if __name__ == "__main__":
    create_project_structure()