const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 3000;
const QUIZ_FILE = path.join(__dirname, "data", "quiz.json");

app.use(express.json());
app.use(express.static("public")); // Serve frontend files

// Load quiz data
app.get("/api/quiz", (req, res) => {
    fs.readFile(QUIZ_FILE, "utf8", (err, data) => {
        if (err) {
            return res.status(500).json({ error: "Failed to load quiz data" });
        }
        res.json(JSON.parse(data));
    });
});

// Save a new question
app.post("/api/quiz", (req, res) => {
    fs.readFile(QUIZ_FILE, "utf8", (err, data) => {
        let quizData = err ? [] : JSON.parse(data);
        quizData.push(req.body);

        fs.writeFile(QUIZ_FILE, JSON.stringify(quizData, null, 2), (err) => {
            if (err) {
                return res.status(500).json({ error: "Failed to save question" });
            }
            res.json({ message: "Question added", quizData });
        });
    });
});

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
