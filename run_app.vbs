Option Explicit
Dim sh, fso, appDir, port, py, q, cmd, i

Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\") - 1)
sh.CurrentDirectory = appDir
port = "8510"    ' own port so it can run next to other Kinentrix builds
' Python: this folder's .venv (made by setup.cmd); falls back to a sibling BetaBeta_260828 venv.
py = appDir & "\.venv\Scripts\python.exe"
If Not fso.FileExists(py) Then py = appDir & "\..\BetaBeta_260828\.venv\Scripts\python.exe"
q = Chr(34)

' ---------------------------------------------------------------- sanity checks
If Not fso.FileExists(appDir & "\models\betabeta_v1.joblib") Then
        MsgBox "Model file not found:" & vbCrLf & _
                      "    models\betabeta_v1.joblib" & vbCrLf & vbCrLf & _
                      "The models folder is missing from this copy.", _
                      vbExclamation, "KINENTRIX Biologics"
        WScript.Quit 1
End If

' ---------------------------------------------------------------- first run: install
' No Python environment yet -> offer to run setup.cmd (creates .venv and installs the
' pinned packages from requirements.txt), then continue with the new .venv.
If Not fso.FileExists(py) Then
        If MsgBox("First run: Python packages are not installed yet." & vbCrLf & vbCrLf & _
                  "Run setup.cmd now? (one time, about 2-5 minutes, needs internet)", _
                  vbYesNo + vbQuestion, "KINENTRIX Biologics") = vbYes Then
                sh.Run "cmd /c " & q & appDir & "\setup.cmd" & q, 1, True   ' wait until done
        End If
        py = appDir & "\.venv\Scripts\python.exe"
        If Not fso.FileExists(py) Then
                MsgBox "Python environment not found." & vbCrLf & vbCrLf & _
                       "Run setup.cmd in this folder once, then start run_app.vbs again.", _
                       vbCritical, "KINENTRIX Biologics"
                WScript.Quit 1
        End If
End If

' ---------------------------------------------------------------- start server
If Not IsPortOpen(port) Then
        cmd = "cmd /c title KINENTRIX Biologics 0.90 Server && " & q & py & q & " -m streamlit run app.py " & _
                    "--server.port " & port & " --server.address 0.0.0.0 --server.headless true" & _
                    " --theme.base dark --theme.primaryColor #38BDF8 --theme.backgroundColor #0B1220 --theme.secondaryBackgroundColor #111C30 --theme.textColor #E2E8F0 --browser.gatherUsageStats false"
        sh.Run cmd, 7, False   ' 7 = minimized, False = don't wait

        For i = 1 To 60
                WScript.Sleep 1000
                If IsPortOpen(port) Then Exit For
        Next
End If

If IsPortOpen(port) Then
        sh.Run "http://localhost:" & port & "/", 1, False
Else
        MsgBox "Server did not start within 60 seconds." & vbCrLf & vbCrLf & _
                      "Try running this manually from a terminal in this folder:" & vbCrLf & _
                      "    .venv\Scripts\python -m streamlit run app.py --server.port " & port, _
                      vbExclamation, "KINENTRIX Biologics"
End If

Function IsPortOpen(p)
        Dim exec, out
        On Error Resume Next
        Set exec = sh.Exec("cmd /c netstat -ano -p tcp | findstr LISTENING | findstr :" & p)
        out = exec.StdOut.ReadAll()
        On Error GoTo 0
        IsPortOpen = (InStr(out, ":" & p) > 0)
End Function
