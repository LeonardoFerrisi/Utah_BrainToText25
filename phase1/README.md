# Utah Neuroengineering B2TXT25': Phase 1

NOTE:
+ This program has been designed to work on both Linux and Windows
    + I attempted to get this working on Docker, however I think this initial setup should be fine for the time being...
+ Make sure that you are in the **phase1** directory
    
## Q&A

+ Why does this need to run on Linux?
    + The baseline people designed their program to run using a `redis-server`
    + This basically allows them to have mulitple programs running in the background
    + There is a way to make a multiple processes, with inter-process communication on any platform, but that takes a lot of extra work...

## Setup (General)

See main README, follow instructions for opening up **Utah_Braintotext25** in a dev container in vs code.

#### Troubleshooting

If the dev container is struggling to run:
- Make sure Docker or Docker Desktop are running