# Utah Neuroengineering B2TXT25': Phase 1

NOTE:
+ This program has been designed to work on both Linux and Windows
    + I attempted to get this working on Docker, however I think this initial setup should be fine for the time being...
+ Make sure that you are in the **phase1** directory
    
## Setup (Windows)

### Installing Windows Subsystem for Linux (WSL)
+ Run `setup_wsl.bat` to install *windows subsystem for linux (wsl)*
    + This will allow you to run everything on your windows machine in an emulated linux environment

+ Once WSL has installed restart you machine.

### Setting up developer environment

Make sure you have the following installed:
+ VSCode

#### Steps
+ Activate WSL in a *terminal* using the following command: `wsl`
+ Change directory to the current one the phase1 folder is located in. (Example: `usr/Documents/UTAH_BRAINTOTEXT25/phase1`)
+ Run the following command to finish setting up the dev environment:

```bash
sudo setup.sh
```


#### Troubleshooting

If trying to run the `setup.sh` script, you may get the error:

    cannot execute: required file not found

If you are using wsl, you may have accidently opened the .sh file on the windows side first.
This will cause CRLF endings to be present in the file.

To fix this, you'll need to install the package `dos2unix` on your linux wsl virtual environment.

###### Steps

```bash
wsl
```

```bash
sudo apt install dos2unix
```

```bash
dos2unix phase1/setup.sh
```

You should see an output like:
```bash
dos2unix: converting file phase1/setup.sh to Unix format...
```

The setup.sh file should now work when you run using

```bash
sudo setup.sh
```