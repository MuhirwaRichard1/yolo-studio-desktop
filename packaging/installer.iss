; Inno Setup script for YOLO Studio.
;
; Build with:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
;
; Expects the PyInstaller bundle at dist\YOLOStudio\.

#define AppName        "YOLO Studio"
; Overridable from the command line: ISCC /DAppVersion=1.1.0 ...
#ifndef AppVersion
  #define AppVersion   "1.0.0"
#endif
#define AppPublisher   "MuhirwaRichard1"
#define AppExeName     "YOLOStudio.exe"
#define AppURL         "https://github.com/MuhirwaRichard1/yolo-studio-desktop"
#define SourceDir      "..\dist\YOLOStudio"

[Setup]
AppId={{6F3B9A21-8E4C-4E7B-9D2A-1C5E7A0B4D93}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\dist\installer
OutputBaseFilename=YOLOStudio-{#AppVersion}-windows-x64-setup
SetupIconFile=icons\yolostudio.ico
UninstallDisplayIcon={app}\{#AppExeName}
WizardStyle=modern
; The payload is several GB of CUDA libraries. Solid LZMA2 buys a lot here, at
; the cost of a slow compress; decompression stays fast.
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
; Per-user install by default so no admin prompt is needed; the user can still
; choose an all-users location if they have rights.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; ~12 GB free is a fair ask: the bundle expands to roughly 6 GB and models are
; downloaded alongside it.
ExtraDiskSpaceRequired=0
DirExistsWarning=no
DisableProgramGroupPage=yes
ShowLanguageDialog=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; PyInstaller writes __pycache__ next to bundled modules at runtime; without
; this the install directory is left behind after uninstall.
Type: filesandordirs; Name: "{app}\_internal\__pycache__"
Type: dirifempty; Name: "{app}"

[Code]
function InitializeSetup(): Boolean;
var
  FreeMB: Cardinal;
  TotalMB: Cardinal;
begin
  Result := True;
  if GetSpaceOnDisk(ExpandConstant('{autopf}'), True, FreeMB, TotalMB) then
  begin
    if FreeMB < 9000 then
    begin
      if MsgBox('YOLO Studio needs about 8 GB of free disk space, but only ' +
                IntToStr(FreeMB) + ' MB appears to be available.' + #13#10#13#10 +
                'Continue anyway?', mbConfirmation, MB_YESNO) = IDNO then
        Result := False;
    end;
  end;
end;
