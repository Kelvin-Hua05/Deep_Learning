# Getting Started with Anaconda

A quick-reference guide for installing Anaconda and managing Python environments and packages with `conda`.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [Managing Environments](#managing-environments)
- [Managing Packages](#managing-packages)
- [Command Cheat Sheet](#command-cheat-sheet)

## Installation and Setup

1. Download and install Anaconda from the [official website](https://www.anaconda.com/download).

2. Activate it. Replace `~/anaconda3` with the directory where you installed Anaconda:

   ```bash
   source ~/anaconda3/bin/activate
   conda init
   ```

3. Restart your terminal. The prompt should now show the active environment:

   ```bash
   (base) max@223ubuntu:~/workspace/machineLearning$
   ```

> **Tip:** To leave the current environment (and return to the plain shell if you are in `base`), run:
>
> ```bash
> conda deactivate
> ```

## Managing Environments

### Create an environment

```bash
conda create --name envName python=3.11
```

### List all environments

```bash
conda env list
```

### Switch to an environment

```bash
conda activate envName
```

The prompt changes from `(base)` to `(envName)`.

### Remove an environment

```bash
conda env remove --name envName
```

## Managing Packages

> Make sure the target environment is activated first, or add `--name envName` to the command.

### Install a package

```bash
conda install packageName
```

### List installed packages

```bash
conda list
```

### Remove a package

```bash
conda remove --name envName packageName
```

## Command Cheat Sheet

| Task                     | Command                                      |
| ------------------------ | -------------------------------------------- |
| Activate base            | `source ~/anaconda3/bin/activate`            |
| Initialize shell         | `conda init`                                 |
| Deactivate environment   | `conda deactivate`                           |
| Create environment       | `conda create --name envName python=3.11`    |
| List environments        | `conda env list`                             |
| Switch environment       | `conda activate envName`                     |
| Install package          | `conda install packageName`                  |
| Remove package           | `conda remove --name envName packageName`    |
| Remove environment       | `conda env remove --name envName`            |
