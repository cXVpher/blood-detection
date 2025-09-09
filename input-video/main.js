const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

let mainWindow;

app.whenReady().then(() => {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    mainWindow.loadFile('index.html');

    ipcMain.handle('select-video', async () => {
        const result = await dialog.showOpenDialog({
            properties: ['openFile'],
            filters: [{ name: 'Videos', extensions: ['mp4', 'avi', 'mkv'] }]
        });

        return result.filePaths[0] || null;
    });

    ipcMain.on('start-yolo', (event, videoPath) => {
        const pythonProcess = spawn('python', ['yolov8.py', videoPath]);

        pythonProcess.stdout.on('data', (data) => {
            console.log(`stdout: ${data}`);
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            console.log(`Process exited with code ${code}`);
        });
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});