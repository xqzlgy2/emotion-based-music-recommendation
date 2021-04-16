$(document).ready(function () {
    let namespace = "/test";
    let video = document.querySelector("#videoElement");
    let canvas = document.querySelector("#canvasElement");
    let ctx = canvas.getContext('2d');
    let photo = document.getElementById('photo');
    let emotion = document.getElementById("emotion");
    let valence = document.getElementById("valence");
    let arousal = document.getElementById("arousal");
    let localMediaStream = null;

    let socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace);

    function sendSnapshot() {
        if (!localMediaStream) {
            return;
        }

        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 300, 150);
        let dataURL = canvas.toDataURL("image/png");
        socket.emit('input image', {image: dataURL});

        socket.on('out-image-event', function (data) {
            photo.setAttribute('src', "data:image/png;base64, " + data.image);
            emotion.innerText = data.results['emotion'];
            valence.innerText = data.results['valence'];
            arousal.innerText = data.results['arousal'];
        });
    }

    socket.on('connect', function () {
        console.log('Connected!');
    });

    socket.on('finished-capturing', function () {
        console.log('finished-capturing');
        sessionStorage.setItem('finished-capturing', 'true');
        window.location.replace('/');
    })

    let constraints = {
        video: {
            width: {min: 640},
            height: {min: 480}
        }
    };

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        video.srcObject = stream;
        localMediaStream = stream;

        setInterval(function () {
            sendSnapshot();
        }, 1000);
    }).catch(function (error) {
        console.log(error);
    });
});