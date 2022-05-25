const express = require('express');
const {spawn} = require('child_process');
const app = express();
const port = 3000;

app.use(express.json());
app.use(express.urlencoded({extended: true}));


app.get('/', (req, res) => {
    res.json({hello: "world"});
})

// 사용자로부터 정보를 입력 받고 추천 노래 리스트를 출력한다.
app.post('/recommend', (req, res) => {
    const {content, image} = req.body
    console.log(content + " : " + image)

    // python 파일 실행을 위한 노력
    let dataToSend;
    const python = spawn('python', ['python_scripts/sentiment_analysis.py', content]);
    python.stdout.on('data', (data) => {
        dataToSend = data.toString();
        console.log(dataToSend)
    })
    // 파이썬 파일이 잘 실행된 경우
    python.on('close', (code) => {
        res.status(200)
        res.send(dataToSend);
    })
})


app.listen(port, () => console.log(port + " request create"));
