$(document).ready(function () {
    // add event listener
    document.getElementById("playlistbtn").addEventListener("click", login);
    document.getElementById("camerabtn").addEventListener("click", checkIfRead);
    document.getElementById("cameranofacebtn").addEventListener("click", checkIfReadNoFace);
    document.getElementById("feedbackbtn").addEventListener("click", finishTest);
    document.getElementById("tryagainbtn").addEventListener("click", tryAgain);

    // request server to get login state
    let signInData = requestServer('/sign_in');

    let ifLogin = signInData.isLogin;
    let ifCaptured = sessionStorage.getItem('finished-capturing') === 'true';
    let ifCapturedNoFace = sessionStorage.getItem('finished-capturing-no-face') === 'true';

    // display capture step if user have logged in
    if (ifLogin) {
        let userName = signInData.content.display_name;
        document.getElementById('title').innerHTML = 'Hi,' + userName + '! Thank you for joining our test.';
        document.getElementById('playlist').style.display = 'none';

        document.getElementById('camera').style.display = ifCaptured ? 'none': 'block';
        document.getElementById('cameranoface').style.display = !ifCaptured || ifCapturedNoFace ? 'none': 'block';
    }
    else {
        document.getElementById('playlist').style.display = 'block';
    }

    // login and capture finished, start recommendation
    if (ifLogin && (ifCaptured || ifCapturedNoFace)) {
        document.getElementById('recommendation').style.display = 'block';
        if (ifCaptured && ifCapturedNoFace) {
            document.getElementById('feedback').style.display = 'block';
        }

        // request server to get recommendation results
        let data = requestServer('/playlists');
        let songs = data.songs;
        let valence = data.detected_valence;

        let tableHtml = '<div class="tableRow"><span class="tableCell text-overflow"><b>Name</b></span>\
            <span class="tableCell text-overflow"><b>Artists</b></span>\
            <span class="tableCell text-overflow"><b>Spotify URL</b></span></div>';

        for (obj of songs) {
            let rowHtml = getRowHtml(obj.name, obj.artists, obj.url);
            tableHtml += rowHtml;
        }

        document.getElementById('recommendationTable').innerHTML = tableHtml;
        document.getElementById('valence').innerHTML = '<b>' + valence + '</b>';
    }

    function getRowHtml(name, artists, url) {
        return '<div class="tableRow"><span class="tableCell text-overflow">' + name +
                 '</span><span class="tableCell text-overflow">' + artists +
                 '</span><span class="tableCell text-overflow"><a href="' + url + '" target="view_window">Click to listen</a></span></div>';
    }

    // get login url and redirect
    function login() {
        let loginUrl = signInData.content;
        window.location.replace(loginUrl);
    }

    // check if user read instructions
    function checkIfRead() {
        let checkBox = document.getElementById('instructionBox');
        if (checkBox.checked) {
            window.location.replace(window.location.protocol + '//' +  window.location.host + '/camera');
        }
        else {
            let warning = document.getElementById('warning');
            warning.innerHTML = 'Please read instructions before continue.';
        }
    }

    // check if user read instructions for second round
    function checkIfReadNoFace() {
        let checkBox = document.getElementById('instructionBoxNoFace');
        if (checkBox.checked) {
            window.location.replace(window.location.protocol + '//' +  window.location.host + '/camera?camera_on=false');
        }
        else {
            let warning = document.getElementById('warningNoFace');
            warning.innerHTML = 'Please read instructions before continue.';
        }
    }

    // jobs after finishing test
    function finishTest() {
        clearUp();
        window.location.replace(surveyLink);
    }

    // jobs before try again
    function tryAgain() {
        clearUp();
        window.location.replace('/');
    }

    function clearUp() {
        requestServer('/sign_out');
        sessionStorage.clear();
    }

    function requestServer(path) {
        let response = $.ajax({
            url: window.location.protocol + '//' +  window.location.host + path,
            async: false
        });
        return response.responseJSON.data;
    }
});
