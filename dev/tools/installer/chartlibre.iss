; ChartLibre's Windows installer, built by dev/tools/build_bundle.py with
; Inno Setup 6 from the ready-made folder (Python and libraries included):
;
;   iscc /DAppVersion=0.1.0 /DSourceDir=...\ChartLibre /DOutputDir=dist
;        /DOutputName=ChartLibre-0.1.0-windows-x86_64-setup chartlibre.iss
;
; It installs for the current user only, under %LOCALAPPDATA%\Programs, so it
; needs no administrator rights - and ChartLibre keeps its settings and logs
; in its own folder, which must therefore be writable (Program Files is not).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #error SourceDir must be given: the ready-made ChartLibre folder
#endif
#ifndef IconFile
  #error IconFile must be given: dev\tools\launcher\chartlibre.ico
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif
#ifndef OutputName
  #define OutputName "ChartLibre-setup"
#endif

[Setup]
AppId={{5B0E4C1D-6A7F-4E2B-9C3D-2F8A1B7E6D40}
AppName=ChartLibre
AppVersion={#AppVersion}
AppPublisher=ChartLibre
AppPublisherURL=https://github.com/ebelland/ChartLibre
DefaultDirName={localappdata}\Programs\ChartLibre
DefaultGroupName=ChartLibre
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#OutputDir}
OutputBaseFilename={#OutputName}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\ChartLibre.exe
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile={#SourceDir}\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ChartLibre"; Filename: "{app}\ChartLibre.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\ChartLibre"; Filename: "{app}\ChartLibre.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\ChartLibre.exe"; Description: "{cm:LaunchProgram,ChartLibre}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; What ChartLibre wrote while it ran: settings, logs, caches. Projects saved
; elsewhere (the home folder, by default) are not touched.
Type: filesandordirs; Name: "{app}\user"
Type: files; Name: "{app}\user.json"
Type: filesandordirs; Name: "{app}\app\logs"
Type: filesandordirs; Name: "{app}\.python"
Type: dirifempty; Name: "{app}"
