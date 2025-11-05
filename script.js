const api = "http://127.0.0.1:8000";
let uploaded = false;
let gaData = null;
let tradData = [];

document.getElementById("uploadBtn").addEventListener("click", uploadFile);
document.getElementById("runGA").addEventListener("click", runGA);
document.getElementById("runTraditional").addEventListener("click", runTraditional);

async function uploadFile() {
    const fileInput = document.getElementById("csvFile");
    const status = document.getElementById("uploadStatus");
    if (!fileInput.files.length) return alert("Choos the CSV file first!!");

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    status.innerText = "Uploading...";
    const res = await fetch(`${api}/upload`, { method: "POST", body: formData });
    const data = await res.json();

    if (data.status === "ok") {
        uploaded = true;
        status.innerText = "Uploaded Successfuly";
    } else {
        status.innerText = "Uploading Failed";
    }
}

async function runGA() {
    if (!uploaded) return alert("Upload the file first!!");

    const gens = parseInt(document.getElementById("generations").value);
    const pop = parseInt(document.getElementById("popSize").value);

    document.getElementById("gaResult").innerText = "Genetic algorithm is running...";
    const params = {
        pop_size: pop,
        generations: gens,
        mut_prob: 0.01,
        crossover_prob: 0.8,
        alpha: 0.9,
        test_size: 0.3,
        random_seed: 42
    };

    const res = await fetch(`${api}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params)
    });
    gaData = await res.json();

    document.getElementById("gaResult").innerHTML = `
    <h4>Genetic algorithm resulte</h4>
    <b>الدقة:</b> ${(gaData.best_accuracy * 100).toFixed(2)}%<br>
    <b>Chosen features:</b> ${gaData.selected_count}<br>
    `;

    drawChart();
}

async function runTraditional() {
    if (!uploaded) return alert("Upload the file first!!");
    const tradDiv = document.getElementById("tradResult");
    tradDiv.innerText = "Trad Method is running..";
    tradData = [];

    const methods = ["lasso", "chi_square", "pca"];

    tradDiv.innerHTML = "";

    for (let method of methods) {
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `<h4>${method.toUpperCase()} ⏳</h4><p>running...</p>`;
        tradDiv.appendChild(card);

        try {
            const res = await fetch(`${api}/run_traditional?method=${method}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ target: "Survived" })
            });
            const data = await res.json();
            tradData.push({ method, ...data });

            card.innerHTML = `
            <h4>Results ${method.toUpperCase()}</h4>
            <b>Accuracy:</b> ${(data.accuracy * 100).toFixed(2)}%<br>
            <b>Number of choosen features</b> ${data.selected_features?.length || data.components || 0}<br>
            <b>Features:</b> ${data.selected_features ? data.selected_features.join(", ") : "—"}
            `;
        } catch (err) {
            card.innerHTML = `<h4>${method.toUpperCase()}</h4><p style="color:red">Error.</p>`;
        }
    }

    drawChart();
}

function drawChart() {
    if (!gaData || tradData.length === 0) return;

    const ctx = document.getElementById("compareChart").getContext("2d");
    if (window.chart) window.chart.destroy();

    const labels = ["Genetic Algorithm", ...tradData.map(t => t.method.toUpperCase())];
    const accuracies = [
        (gaData.best_accuracy * 100).toFixed(2),
        ...tradData.map(t => (t.accuracy * 100).toFixed(2))
    ];

    window.chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Accuracy (%)",
                             data: accuracies,
                             backgroundColor: ["#44bd32", "#487eb0", "#e1b12c", "#9c88ff"]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: "bottom" },
                title: { display: true, text: "Comparison" }
            },
            scales: { y: { beginAtZero: true } }
        }
    });
}
