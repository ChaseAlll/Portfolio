# Table of Contents 

### [1.](#introduction) Introduction
### [2.](#prerequisites) Prerequisites
### [3.](#installing-emulator-and-dependencies) Installing Emulator and Dependencies
 1. ##### Download Resources
 
 
 2. ##### Install Resources
 
### [4.](#configuring-emulator) Configuring Emulator
 1. ##### Map Controller
 
 
 2. ##### Set Game Path
### [5.](#faq-2) FAQ


## Introduction
Dolphin is a flexible and accessible program that emulates the GameCube and Wii consoles. This document will give an in-depth
walkthrough on the first-time installation and configuration of the emulator on a Windows 64-bit machine. 
This documentation is intended for individuals who have little experience with emulators in general or have never installed
this emulator before. Anyone has experience with this emulator or is going to use a version that is different than
the one used should look for information more curated to their use case. Once read, this documentation will give the user the
knowledge to install the emulator and to configure it to use the settings most efficient for a general use case.

## Prerequisites
 * System Requirements
    * OS: 64-bit edition of Windows (7 SP1 or higher), Linux, or macOS (10.10 Yosemite or higher). Windows Vista SP2 and unix-like systems other than Linux are not officially supported but might work.
   * Processor: A CPU with SSE2 support. A modern CPU (3 GHz and Dual Core, not older than 2008) is highly recommended.
   * Graphics: A reasonably modern graphics card (Direct3D 10.0 / OpenGL 3.0). A graphics card that supports Direct3D 11 / OpenGL 4.4 is recommended.
 * Rom’s to Run on the emulator in their own dedicated folder
 * Connection to the Internet
 * A Controller Compatible to Your System
 * 7zip or Other Compression Software Capable of Extracting .7z files
 
## Installing Emulator and Dependencies
 * Navigate to the Dolphin [homepage](https://dolphin-emu.org/)
 ![DolphinHomePage](https://raw.githubusercontent.com/RobertGageStroud/Portfolio/master/P2Pictures/DolphinWebsite.png)

 * Click either download link on the page
 * Download the latest version of Dolphin (Current - 5.0-11827)
 * Navigate to the Visual Studio C++ Redistributable [page](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)
 * Download Visual Studio 2015, 2017, and 2019 based on your system.
  
 * Navigate to where you downloaded the above files
 * Run the vc_redist.x--.exe file
   * If it fails it means it was already installed on your system
 * Extract Dolphin from its .7z file to where you want the program
 
## Configuring Emulator
 * Go into the extracted folder and run the Dolphin Program.
 * Click the Controllers Button
 ![DolphinStart](https://raw.githubusercontent.com/RobertGageStroud/Portfolio/master/P2Pictures/DolphinHome.png)
 * Under GameCube Controllers, Port1, select Standard Controller from the drop-down menu.
 ![DolphinControllerSelect](https://raw.githubusercontent.com/RobertGageStroud/Portfolio/master/P2Pictures/ControllerSettings.png)
 * Click Configure
 * Ensure your Controller is connected. 
 * Under Device select Xinput/… from the drop-down menu
 ![DolphincontrollerMapping](https://raw.githubusercontent.com/RobertGageStroud/Portfolio/master/P2Pictures/ControllerMapping.png)
   * If Xinput is not displayed, ensure controller is connected and click refresh.
    

 * Map controller inputs to your preference. 
 * Under Profile give this controller a name. 
 * Save the profile
 * Close the both controller windows 
 * Click the Config button
 ![DolphinStart](https://raw.githubusercontent.com/RobertGageStroud/Portfolio/master/P2Pictures/DolphinHome.png)
* Navigate to the Paths tab
![DolphinPaths](https://raw.githubusercontent.com/RobertGageStroud/Portfolio/master/P2Pictures/Paths.png)
 * Click add
 * Navigate and select folder where Rom’s are stored. 
  * If rom’s are sorted in someway into different folders. Ensure they are all put into a single roms folder and check the search subfolder option. 
 * Click close
 * Choose a Rom to run and play it!

## FAQ
 * Are there any performance settings that need changed?
   * New versions of Dolphin come with the most optimized settings preselected. If you are having trouble, try lowering the resolution or choosing a different renderer. 
 * "This application has failed to start because XINPUT1_3.dll was not found"
   * This only occurs on systems using an older version of Windows (below 10). Like how we installed the VS C++ redistributable install the Latest DirectX Runtime from [here](https://support.microsoft.com/en-us/help/179113/how-to-install-the-latest-version-of-directx) 
 * “Im using a PS3/4 controller and it is not recognized.”
   * Install the SCPToolkit from [here](https://github.com/nefarius/ScpToolkit) this will allow windows to recognize the controller as a Bluetooth device and retry adding the controller. 

