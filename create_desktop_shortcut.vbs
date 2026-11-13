Option Explicit
' Creates (or refreshes) the VisionAI Sentinel desktop shortcut for this copy
' of the project. Safe to run again after moving the folder.

Dim fso, shell, projectDir, desktopDir, shortcutPath, link
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
desktopDir = shell.SpecialFolders("Desktop")
shortcutPath = fso.BuildPath(desktopDir, "VisionAI Sentinel.lnk")

Set link = shell.CreateShortcut(shortcutPath)
link.TargetPath = fso.BuildPath(projectDir, "start_app.bat")
link.WorkingDirectory = projectDir
link.IconLocation = fso.BuildPath(projectDir, "visionai_icon.ico") & ",0"
link.WindowStyle = 1
link.Description = "Launch the VisionAI Sentinel AI camera command center"
link.Save

If Not WScript.Arguments.Named.Exists("quiet") Then
    MsgBox "Desktop shortcut created:" & vbCrLf & shortcutPath, vbInformation, "VisionAI Sentinel"
End If
