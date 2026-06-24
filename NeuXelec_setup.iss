#define MyAppName      "NeuXelec"
#define MyAppVersion   "1.0.0"
#define MyAppPublisher "Jules Veber - HUG Geneva"
#define MyAppURL       "https://neuxelec.com"
#define MyAppExeName   "NeuXelec.exe"

[Setup]
AppId={{8CE5A64A-3C92-4C25-9427-E485340F8121}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
AppComments=SEEG electrode reconstruction and 3D neuroimaging visualization

; Installation directory - user-level, no admin required
DefaultDirName={localappdata}\Programs\NeuXelec
DefaultGroupName=NeuXelec
DisableProgramGroupPage=yes

; Output
OutputDir=installer
OutputBaseFilename=NeuXelec_Setup_{#MyAppVersion}

; Icons
SetupIconFile=resources\images\brain_logo.ico
UninstallDisplayIcon={app}\NeuXelec.exe
UninstallDisplayName=NeuXelec {#MyAppVersion}

; License shown during installation (must be accepted)
LicenseFile=LICENSE

; Compression
Compression=lzma2/ultra64
SolidCompression=yes

; Windows / architecture
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; No admin rights needed
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Wizard appearance
WizardStyle=modern
DisableReadyPage=no
DisableWelcomePage=no

; Misc
CloseApplications=yes
RestartIfNeededByRun=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french";  MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\NeuXelec\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start menu
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
; Desktop (optional, controlled by task above)
Name: "{autodesktop}\{#MyAppName}";  Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove any files created by the app at runtime (project files saved in app dir)
Type: filesandordirs; Name: "{app}"

[Code]
// Show a simple research-use reminder at the end of the wizard
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssDone then
    MsgBox(
      'NeuXelec has been installed successfully.' + #13#10 + #13#10 +
      'Important: NeuXelec is a research tool only.' + #13#10 +
      'It is not a certified medical device and must not be used' + #13#10 +
      'for clinical diagnosis or treatment decisions.',
      mbInformation, MB_OK
    );
end;
