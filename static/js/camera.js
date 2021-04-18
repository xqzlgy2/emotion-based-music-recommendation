$(document).ready(function () {
    let namespace = "/test";
    let video = document.getElementById("videoElement");
    let canvas = document.getElementById("canvasElement");
    let ctx = canvas.getContext('2d');
    let photo = document.getElementById('photo');
    let emotion = document.getElementById("emotion");
    let valence = document.getElementById("valence");
    let arousal = document.getElementById("arousal");
    let captured = document.getElementById("captured");
    let localMediaStream = null;
    const urlParams = new URLSearchParams(window.location.search);
    const show_face = urlParams.get("camera_on");

    if (show_face === 'false') {
        document.getElementById("container").style.display = 'none';
    }

    let socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace);
    let count_frame = 0;

    function sendSnapshot() {
        if (!localMediaStream) {
            return;
        }

        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 300, 150);
        let dataURL = canvas.toDataURL("image/png");
        socket.emit('input image', {image: dataURL});

    }

    socket.on('connect', function () {
        console.log('Connected!');
    });

    socket.on('out-image-event', function (data) {
        count_frame += 1;
        captured.innerText = `${count_frame}/10 frames recorded`;
        photo.setAttribute('src', "data:image/png;base64, " + data.image);
        emotion.innerText = `Emotion: ${data.results['emotion']}`;
        valence.innerText = `Valence: ${data.results['valence']}`;
        arousal.innerText = `Arousal: ${data.results['arousal']}`;
    });

    socket.on('finished-capturing', function () {
        console.log('finished-capturing');
        count_frame = 0;
        if (show_face === 'false') {
            sessionStorage.setItem('finished-capturing-no-face', 'true');
        }
        else {
            sessionStorage.setItem('finished-capturing', 'true');
        }
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
