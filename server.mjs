import express from 'express';
import { MongoClient } from 'mongodb';
import { ChartJSNodeCanvas } from 'chartjs-node-canvas';
import bodyParser from 'body-parser';
import fetch from 'node-fetch';

const config = {
    mongoUri: 'mongodb+srv://username:password@cluster0.bptf95x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
    dbName: 'CSV_Analyzer',
    collectionName: 'CSV_Analyzer2',
    geminiApiKey: 'your key'
};

const app = express();
const PORT = 3000;
app.use(bodyParser.json());

const client = new MongoClient(config.mongoUri);
const chartCanvas = new ChartJSNodeCanvas({ width: 800, height: 400 });

// Route: Home page with input form
app.get('/', (req, res) => {
    res.send(`
        <html>
        <head><title>LLM Chart Generator</title></head>
        <body>
            <h2>🔍 Ask a Question to Visualize Sales</h2>
            <form onsubmit="handleSubmit(event)">
                <input type="text" id="question" placeholder="e.g., Compare sales amount by region" size="50" />
                <button type="submit">Generate</button>
            </form>
            <br/>
            <img id="chart" style="max-width:700px;" />
            <script>
                async function handleSubmit(event) {
                    event.preventDefault();
                    const question = document.getElementById("question").value;
                    const res = await fetch("/query", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question })
                    });
                    const blob = await res.blob();
                    document.getElementById("chart").src = URL.createObjectURL(blob);
                }
            </script>
        </body>
        </html>
    `);
});

// MongoDB fetch
async function fetchData() {
    await client.connect();
    const collection = client.db(config.dbName).collection(config.collectionName);
    return await collection.find().toArray();
}

// Group and sum sales
function groupByAndSum(data, xKey) {
    const result = {};
    data.forEach(item => {
        const key = item[xKey];
        const amount = parseFloat(item.sales_amount || 0);
        if (!result[key]) result[key] = 0;
        result[key] += amount;
    });
    return result;
}

// Gemini API to get x-axis field
async function getXAxisField(query, schemaKeys) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${config.geminiApiKey}`;
    const headers = { "Content-Type": "application/json" };
    const payload = {
        contents: [
            {
                parts: [
                    {
                        text: `You are a data assistant.\nGiven the fields: ${schemaKeys.join(', ')}\nY-axis is fixed: "sales_amount".\nFrom this query: "${query}", return the best X-axis field to group by.\nOnly return the field name.`
                    }
                ]
            }
        ]
    };
    const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Gemini API error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    const textResponse = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    return textResponse.trim().replace(/[`'"]/g, '');
}

// Chart route
app.post('/query', async (req, res) => {
    try {
        const { question } = req.body;
        const data = await fetchData();
        const schema = Object.keys(data[0]);
        const xAxisField = await getXAxisField(question, schema);
        const grouped = groupByAndSum(data, xAxisField);

        const labels = Object.keys(grouped);
        const values = Object.values(grouped);

        const chartConfig = {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Total Sales Amount',
                    data: values,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)'
                }]
            },
            options: {
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Sales Amount'
                        },
                        beginAtZero: true
                    },
                    x: {
                        title: {
                            display: true,
                            text: xAxisField
                        }
                    }
                }
            }
        };

        const image = await chartCanvas.renderToBuffer(chartConfig);
        res.set('Content-Type', 'image/png');
        res.send(image);

    } catch (err) {
        console.error(err);
        res.status(500).send('Failed to generate chart');
    }
});

app.listen(PORT, () => {
    console.log(`✅ Server running at http://localhost:${PORT}`);
});
