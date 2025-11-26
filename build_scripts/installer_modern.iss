; ==================================================================================
; Retro ML Trainer - Modern Installer Script
; ==================================================================================
; This is a modernized Inno Setup installer with custom UI, GPU detection,
; and enhanced user experience.
;
; Requirements:
;   - Inno Setup 6.0 or later
;   - Custom images in build_scripts/installer_images/ (optional)
; ==================================================================================

#define MyAppName "Retro ML Trainer"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Retro ML Project"
#define MyAppURL "https://github.com/StevenVC-P/youtube-10-ml-display"
#define MyAppExeName "RetroMLTrainer.exe"
#define MyAppDescription "Train AI to master classic Atari games"

[Setup]
; ==================================================================================
; App Information
; ==================================================================================
AppId={{A7B5C8D9-1234-5678-90AB-CDEF12345678}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
AppCopyright=Copyright (C) 2025 {#MyAppPublisher}
AppComments={#MyAppDescription}

; ==================================================================================
; Installation Directories
; ==================================================================================
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableDirPage=no
DisableReadyPage=no

; ==================================================================================
; Output Configuration
; ==================================================================================
OutputDir=..\installer_output
OutputBaseFilename=RetroMLTrainer-Setup-v{#MyAppVersion}
SetupIconFile=installer_images\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

; ==================================================================================
; Compression & Performance
; ==================================================================================
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4

; ==================================================================================
; Modern UI Configuration
; ==================================================================================
WizardStyle=modern
WizardSizePercent=120,100
WizardImageFile=installer_images\wizard_large.bmp
WizardSmallImageFile=installer_images\wizard_small.bmp

; Modern colors - Dark blue theme
WizardImageBackColor=$2B2B2B
SetupIconFile=installer_images\app_icon.ico

; ==================================================================================
; Privileges & Security
; ==================================================================================
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; ==================================================================================
; Architecture
; ==================================================================================
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; ==================================================================================
; Version Information
; ==================================================================================
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} Setup
VersionInfoCopyright=Copyright (C) 2025
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
; Custom messages for better UX
WelcomeLabel1=Welcome to [name] Setup
WelcomeLabel2=This will install [name/ver] on your computer.%n%nTrain AI agents to play classic Atari games using state-of-the-art reinforcement learning!%n%nRecommended: NVIDIA GPU with CUDA support for best performance.
FinishedHeadingLabel=Setup Complete - Ready to Train AI!
FinishedLabelNoIcons=Setup has successfully installed {#MyAppName}.%n%nOn first launch, the setup wizard will guide you through:%n  • ML dependencies installation (PyTorch + CUDA)%n  • GPU detection and configuration%n  • Atari ROM installation%n  • Storage configuration
FinishedLabel=Setup has successfully installed [name].%n%nClick Finish to close Setup and get started!
ButtonFinish=&Launch Application
ClickFinish=Click Finish to exit Setup and start training AI!

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"
Name: "quicklaunchicon"; Description: "Create a &Quick Launch shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; ==================================================================================
; Main Application Files
; ==================================================================================
Source: "..\dist\RetroMLTrainer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; ==================================================================================
; Documentation
; ==================================================================================
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\docs\USER_GUIDE.md"; DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\docs\DISTRIBUTION_GUIDE.md"; DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
; ==================================================================================
; Shortcuts
; ==================================================================================
; Start Menu
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "Train AI to play Atari games"
Name: "{group}\User Guide"; Filename: "{app}\docs\USER_GUIDE.md"; Flags: skipifsourcedoesntexist
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

; Desktop icon
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "Train AI to play Atari games"

; Quick Launch icon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
; ==================================================================================
; Post-Installation Actions
; ==================================================================================
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; ==================================================================================
; Cleanup (commented out by default - user data preservation)
; ==================================================================================
; Uncomment to delete user data on uninstall (use with caution!)
; Type: filesandordirs; Name: "{userappdata}\RetroMLTrainer"
; Type: filesandordirs; Name: "{localappdata}\RetroMLTrainer"

[Code]
; ==================================================================================
; CUSTOM PASCAL CODE
; ==================================================================================

var
  GPUDetectionPage: TOutputMsgWizardPage;
  SystemInfoPage: TOutputMsgWizardPage;
  GPUDetected: Boolean;
  GPUName: String;
  HasNVIDIAGPU: Boolean;

// ==================================================================================
// GPU Detection Functions
// ==================================================================================

function CheckForNVIDIAGPU(): Boolean;
var
  ResultCode: Integer;
  GPUOutput: AnsiString;
  TempFile: String;
begin
  Result := False;
  GPUName := 'Not detected';

  // Try to detect NVIDIA GPU using nvidia-smi
  TempFile := ExpandConstant('{tmp}\gpu_check.txt');

  // Method 1: nvidia-smi
  if Exec('nvidia-smi', '--query-gpu=name --format=csv,noheader', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if ResultCode = 0 then
    begin
      Result := True;
      GPUName := 'NVIDIA GPU (detected)';
      Exit;
    end;
  end;

  // Method 2: Check registry for NVIDIA drivers
  if RegKeyExists(HKEY_LOCAL_MACHINE, 'SOFTWARE\NVIDIA Corporation\Global') or
     RegKeyExists(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Services\nvlddmkm') then
  begin
    Result := True;
    GPUName := 'NVIDIA GPU (from registry)';
    Exit;
  end;

  // Method 3: Check for CUDA installation
  if RegKeyExists(HKEY_LOCAL_MACHINE, 'SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA') then
  begin
    Result := True;
    GPUName := 'NVIDIA GPU (CUDA installed)';
    Exit;
  end;
end;

// ==================================================================================
// System Information Functions
// ==================================================================================

function GetWindowsVersionString(): String;
var
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  Result := 'Windows ' + IntToStr(Version.Major) + '.' + IntToStr(Version.Minor);

  // Add friendly names
  if Version.Major = 10 then
  begin
    if Version.Build >= 22000 then
      Result := 'Windows 11'
    else
      Result := 'Windows 10';
  end;
end;

function GetTotalRAM(): String;
var
  MemoryStatus: TMemoryStatusEx;
  TotalGB: Extended;
begin
  MemoryStatus.dwLength := SizeOf(MemoryStatus);
  if GlobalMemoryStatusEx(MemoryStatus) then
  begin
    TotalGB := MemoryStatus.ullTotalPhys / (1024 * 1024 * 1024);
    Result := Format('%.1f GB', [TotalGB]);
  end
  else
    Result := 'Unknown';
end;

function GetAvailableDiskSpace(Path: String): String;
var
  FreeBytes, TotalBytes: Int64;
  FreeGB: Extended;
begin
  if GetSpaceOnDisk64(Path, FreeBytes, TotalBytes) then
  begin
    FreeGB := FreeBytes / (1024 * 1024 * 1024);
    Result := Format('%.1f GB', [FreeGB]);
  end
  else
    Result := 'Unknown';
end;

// ==================================================================================
// Custom Wizard Pages
// ==================================================================================

procedure CreateSystemInfoPage();
var
  InfoText: String;
begin
  SystemInfoPage := CreateOutputMsgPage(
    wpWelcome,
    'System Information',
    'Checking your system capabilities',
    ''
  );

  InfoText := 'System Requirements Check:' + #13#10 + #13#10;
  InfoText := InfoText + 'Operating System: ' + GetWindowsVersionString() + #13#10;
  InfoText := InfoText + 'Total RAM: ' + GetTotalRAM() + #13#10;
  InfoText := InfoText + 'Available Disk Space: ' + GetAvailableDiskSpace(ExpandConstant('{sd}')) + #13#10;
  InfoText := InfoText + #13#10;

  // GPU Detection
  HasNVIDIAGPU := CheckForNVIDIAGPU();

  if HasNVIDIAGPU then
  begin
    InfoText := InfoText + 'GPU: ' + GPUName + ' ✓' + #13#10;
    InfoText := InfoText + #13#10;
    InfoText := InfoText + 'Great! Your NVIDIA GPU will enable fast AI training.' + #13#10;
    InfoText := InfoText + 'PyTorch with CUDA support will be installed on first run (~2.5 GB).' + #13#10;
  end
  else
  begin
    InfoText := InfoText + 'GPU: No NVIDIA GPU detected' + #13#10;
    InfoText := InfoText + #13#10;
    InfoText := InfoText + 'Note: Training will use CPU mode (slower).' + #13#10;
    InfoText := InfoText + 'For best performance, an NVIDIA GPU is recommended.' + #13#10;
    InfoText := InfoText + #13#10;
    InfoText := InfoText + 'PyTorch CPU version will be installed on first run (~800 MB).' + #13#10;
  end;

  InfoText := InfoText + #13#10;
  InfoText := InfoText + 'Installation Requirements:' + #13#10;
  InfoText := InfoText + '  • Windows 10/11 (64-bit) ✓' + #13#10;
  InfoText := InfoText + '  • 150 MB for initial installation' + #13#10;
  InfoText := InfoText + '  • 10+ GB recommended for training data' + #13#10;
  InfoText := InfoText + '  • Internet connection (for first-run setup)' + #13#10;

  SystemInfoPage.RichEditViewer.RTFText := '{\rtf1\ansi\deff0' + InfoText + '}';
end;

// ==================================================================================
// Installation Event Handlers
; ==================================================================================

function InitializeSetup(): Boolean;
var
  Version: TWindowsVersion;
  ErrorMessage: String;
begin
  Result := True;

  // Check Windows version
  GetWindowsVersionEx(Version);

  if Version.Major < 10 then
  begin
    ErrorMessage := 'This application requires Windows 10 or later.' + #13#10 + #13#10;
    ErrorMessage := ErrorMessage + 'Current Windows version: ' + IntToStr(Version.Major) + '.' + IntToStr(Version.Minor) + #13#10;
    ErrorMessage := ErrorMessage + #13#10 + 'Please upgrade your operating system to continue.';
    MsgBox(ErrorMessage, mbError, MB_OK);
    Result := False;
    Exit;
  end;

  // Check disk space (at least 5 GB for safe installation)
  if GetSpaceOnDisk(ExpandConstant('{sd}'), False, 0, 0) < (5 * 1024 * 1024 * 1024) then
  begin
    ErrorMessage := 'Insufficient disk space!' + #13#10 + #13#10;
    ErrorMessage := ErrorMessage + 'This application requires at least 5 GB of free disk space.' + #13#10;
    ErrorMessage := ErrorMessage + 'Please free up some space and try again.';
    MsgBox(ErrorMessage, mbError, MB_OK);
    Result := False;
    Exit;
  end;
end;

procedure InitializeWizard();
begin
  // Create custom pages
  CreateSystemInfoPage();
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  // Update system info when page is shown
  if CurPageID = SystemInfoPage.ID then
  begin
    // Refresh system information
    CreateSystemInfoPage();
  end;

  // Show helpful message on finish page
  if CurPageID = wpFinished then
  begin
    // Could add additional logic here
  end;
end;

// ==================================================================================
// Upgrade Detection
// ==================================================================================

function GetUninstallString(): String;
var
  sUnInstPath: String;
  sUnInstallString: String;
begin
  sUnInstPath := ExpandConstant('Software\Microsoft\Windows\CurrentVersion\Uninstall\{#emit SetupSetting("AppId")}_is1');
  sUnInstallString := '';
  if not RegQueryStringValue(HKLM, sUnInstPath, 'UninstallString', sUnInstallString) then
    RegQueryStringValue(HKCU, sUnInstPath, 'UninstallString', sUnInstallString);
  Result := sUnInstallString;
end;

function IsUpgrade(): Boolean;
begin
  Result := (GetUninstallString() <> '');
end;

function UnInstallOldVersion(): Integer;
var
  sUnInstallString: String;
  iResultCode: Integer;
begin
  Result := 0;
  sUnInstallString := GetUninstallString();
  if sUnInstallString <> '' then
  begin
    sUnInstallString := RemoveQuotes(sUnInstallString);
    if Exec(sUnInstallString, '/SILENT /NORESTART /SUPPRESSMSGBOXES', '', SW_HIDE, ewWaitUntilTerminated, iResultCode) then
      Result := 3
    else
      Result := 2;
  end
  else
    Result := 1;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssInstall) then
  begin
    if IsUpgrade() then
    begin
      // Uninstall old version
      UnInstallOldVersion();
    end;
  end;
end;

// ==================================================================================
// Custom Uninstall
// ==================================================================================

function InitializeUninstall(): Boolean;
begin
  Result := True;

  if MsgBox('Do you want to keep your training data and videos?', mbConfirmation, MB_YESNO) = IDYES then
  begin
    // User wants to keep data - just uninstall the app
    Result := True;
  end
  else
  begin
    // User wants to delete everything
    Result := MsgBox('This will delete ALL training data, models, and videos. Are you sure?', mbConfirmation, MB_YESNO) = IDYES;
  end;
end;

// ==================================================================================
// END OF SCRIPT
// ==================================================================================
