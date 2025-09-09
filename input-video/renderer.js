const { ipcRenderer } = require('electron');

document.getElementById('select-video').addEventListener('click', async () => {
    const videoPath = await ipcRenderer.invoke('select-video');
    if (videoPath) {
        document.getElementById('video-path').textContent = `Selected: ${videoPath}`;
        document.getElementById('start-detection').disabled = false;
    }
});

document.getElementById('start-detection').addEventListener('click', () => {
    const videoPath = document.getElementById('video-path').textContent.replace('Selected: ', '');
    ipcRenderer.send('start-yolo', videoPath);
});