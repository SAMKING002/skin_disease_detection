document.addEventListener("DOMContentLoaded", function () {
    

    // Labels for all charts
    const diseaseLabels = [
        "Acne & Rosacea", "Actinic Keratosis & Malignant Lesions", "Atopic Dermatitis", "Bullous Disease",
        "Bacterial Infections", "Eczema", "Exanthems & Drug Eruptions", "Hair Loss & Hair Diseases",
        "Herpes, HPV & STDs", "Light Disorders", "Lupus & Connective Tissue Diseases", "Melanoma & Moles",
        "Nail Fungus & Nail Diseases", "Poison Ivy & Contact Dermatitis", "Psoriasis & Related Diseases",
        "Scabies & Infestations", "Seborrheic Keratoses & Benign Tumors", "Systemic Disease", "Fungal Infections",
        "Urticaria (Hives)", "Vascular Tumors", "Vasculitis", "Warts & Viral Infections"
    ];

    // Dummy data for number of cases
    const dummyData = [
        50, 30, 90, 90, 30, 50, 70, 13, 16, 11, 79, 20, 
        85, 30, 95, 27, 35, 31, 55, 14, 10, 27
    ];

    // Generate different colors for each label
    const barColors = [
        "#FF5733", "#33FF57", "#3357FF", "#FF33A8", "#33FFF5", "#A833FF",
        "#FF9F33", "#33A1FF", "#FF3333", "#33FF85", "#AA33FF", "#33FFC1",
        "#FF6A33", "#3381FF", "#FF3385", "#33FFAA", "#AAFF33", "#6A33FF",
        "#FFAA33", "#339FFF", "#FF337A", "#33E5FF"
    ];

    // Bar Chart
    const barCtx = document.getElementById("barChart").getContext("2d");
    new Chart(barCtx, {
        type: "bar",
        data: {
            labels: diseaseLabels,
            datasets: [{
                label: "Number of Cases",
                data: dummyData,
                backgroundColor: barColors,
                borderColor: barColors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: "Skin Disease Cases by Category",
                    font: { size: 18, weight: "bold" },
                    color: "#333",
                    padding: 20
                },
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (tooltipItems) => diseaseLabels[tooltipItems[0].dataIndex],
                        label: (tooltipItem) => `Cases: ${tooltipItem.raw}`
                    }
                }
            },
            scales: {
                x: { display: false }, // Hide x-axis labels
                y: { beginAtZero: true }
            }
        }
    });

    // Pie Chart
    const pieCtx = document.getElementById("pieChart").getContext("2d");
    new Chart(pieCtx, {
        type: "pie",
        data: {
            labels: diseaseLabels,
            datasets: [{
                data: dummyData,
                backgroundColor: barColors
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: "Percentage of Skin Disease Cases",
                    font: { size: 18, weight: "bold" },
                    color: "#333",
                    padding: 20
                },
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (tooltipItems) => diseaseLabels[tooltipItems[0].dataIndex],
                        label: (tooltipItem) => `Cases: ${tooltipItem.raw}`
                    }
                }
            }
        }
    });

    // Line Chart
    const lineCtx = document.getElementById("lineChart").getContext("2d");
    new Chart(lineCtx, {
        type: "line",
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            datasets: [{
                label: "Signal Frequency Distribution",
                data: [3, 5, 8, 6, 10, 7, 11, 9, 15, 13, 14, 12],
                borderColor: "rgba(153, 102, 255, 1)",
                backgroundColor: "rgba(153, 102, 255, 0.2)",
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: "Signal Frequency Distribution Over the Months",
                    font: { size: 18, weight: "bold" },
                    color: "#333",
                    padding: 20
                }
            },
            scales: { y: { beginAtZero: true } }
        }
    });

    // Scroll-Up Button
    const scrollUpBtn = document.getElementById("scrollUpBtn");
    window.addEventListener("scroll", function () {
        if (window.scrollY > 200) {
            scrollUpBtn.classList.add("show");
        } else {
            scrollUpBtn.classList.remove("show");
        }
    });

    scrollUpBtn.addEventListener("click", function () {
        window.scrollTo({ top: 0, behavior: "smooth" });
    });
});
document.getElementById("resultContainer").style.display = "block";
const imageElement = document.getElementById("uploadedImage");
imageElement.src = "uploads/" + uploadedFilename; // Ensure this is correct
imageElement.style.display = "block";
fetch("/predict", {
    method: "POST",
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.error) {
        alert("Error: " + data.error);
    } else {
        document.getElementById("resultContainer").style.display = "block";
        document.getElementById("diseaseName").innerText = data.disease;
        document.getElementById("confidenceScore").innerText = data.confidence;
        document.getElementById("uploadedImage").src = data.image_url; // Use returned URL
        document.getElementById("uploadedImage").style.display = "block";
    }
});
