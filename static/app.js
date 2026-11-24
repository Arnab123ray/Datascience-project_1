const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const isLocalDevelopmentHost = ["localhost", "127.0.0.1"].includes(window.location.hostname);
const apiBaseUrl =
  isLocalDevelopmentHost && window.location.port && window.location.port !== "5000"
    ? `${window.location.protocol}//${window.location.hostname}:5000`
    : "";
const fileInput = document.querySelector("#csv-file");
const uploadForm = document.querySelector("#upload-form");
const uploadButton = uploadForm.querySelector("button");
const customerSelect = document.querySelector("#customer-select");
const statusMessage = document.querySelector("#status-message");
let currentFile = null;
let currentSource = "Sample dataset";
let requestNumber = 0;

function apiUrl(path) {
  return `${apiBaseUrl}${path}`;
}

document.querySelector("#sample-download").href = apiUrl("/sample.csv");

function showStatus(message, kind) {
  statusMessage.textContent = message;
  statusMessage.className = `status-message ${kind}`;
  statusMessage.hidden = false;
}

function hideStatus() {
  statusMessage.hidden = true;
  statusMessage.textContent = "";
}

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function money(value) {
  return value == null ? "Not available" : currency.format(value);
}

function segmentClass(segment) {
  return String(segment || "").toLowerCase();
}

function renderRecommendations(recommendations) {
  const container = document.querySelector("#recommendation-list");
  container.replaceChildren();
  if (!recommendations || recommendations.length === 0) {
    const empty = document.createElement("span");
    empty.className = "empty-recommendation";
    empty.textContent = "Not enough customer history for recommendations.";
    container.append(empty);
    return;
  }

  recommendations.forEach((product) => {
    const item = document.createElement("div");
    item.className = "recommendation-item";
    const title = document.createElement("strong");
    title.textContent = product.product_name;
    title.title = product.product_name;
    const detail = document.createElement("span");
    detail.textContent = `${product.category} · ★ ${product.rating ?? "—"}`;
    item.append(title, detail);
    container.append(item);
  });
}

function renderCategories(categories) {
  const chart = document.querySelector("#category-chart");
  chart.replaceChildren();
  const topCategories = categories.slice(0, 6);
  const maximum = Math.max(...topCategories.map((item) => item.revenue || 0), 1);
  topCategories.forEach((category) => {
    const row = document.createElement("div");
    row.className = "category-row";
    const label = document.createElement("span");
    label.className = "category-label";
    label.textContent = category.label;
    label.title = category.label;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(2, (category.revenue / maximum) * 100)}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "category-value";
    value.textContent = money(category.revenue);
    row.append(label, track, value);
    chart.append(row);
  });
  if (topCategories.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-recommendation";
    empty.textContent = "No category data to display.";
    chart.append(empty);
  }
}

function renderSalesTrend(trend) {
  const chart = document.querySelector("#sales-chart");
  chart.replaceChildren();
  if (!trend || trend.length === 0) {
    const empty = document.createElement("span");
    empty.className = "empty-recommendation";
    empty.textContent = "Add a purchase_date column to show sales over time.";
    chart.append(empty);
    return;
  }

  const periods = trend.slice(-6);
  const maximum = Math.max(...periods.map((period) => period.revenue || 0), 1);
  periods.forEach((period) => {
    const column = document.createElement("div");
    column.className = "sales-column";
    column.title = `${period.label}: ${money(period.revenue)}`;
    const bar = document.createElement("span");
    bar.className = "sales-bar";
    bar.style.height = `${Math.max(4, (period.revenue / maximum) * 36)}px`;
    const label = document.createElement("small");
    label.textContent = new Date(`${period.label}-01T00:00:00Z`).toLocaleDateString(
      "en-US",
      { month: "short", timeZone: "UTC" }
    );
    column.append(bar, label);
    chart.append(column);
  });
}

function renderCustomers(customers, selectedId) {
  const table = document.querySelector("#customer-table");
  table.replaceChildren();
  [...customers]
    .sort((left, right) => right.spending - left.spending)
    .forEach((customer) => {
      const row = document.createElement("tr");
      const values = [
        { value: customer.customer_id, className: "table-customer" },
        { value: money(customer.spending) },
        { value: money(customer.predicted_spending) },
        { value: `${customer.purchase_probability}%` },
      ];
      values.forEach((item) => {
        const cell = document.createElement("td");
        cell.textContent = item.value;
        if (item.className) cell.className = item.className;
        row.append(cell);
      });
      const segmentCell = document.createElement("td");
      const pill = document.createElement("span");
      pill.className = `table-segment ${segmentClass(customer.segment)}`;
      pill.textContent = customer.segment;
      segmentCell.append(pill);
      row.append(segmentCell);
      if (customer.customer_id === selectedId) row.classList.add("selected-row");
      table.append(row);
    });
}

function renderDashboard(data) {
  const { overview, model_metrics: metrics, selected_customer: selected } = data;
  setText("#data-caption", currentSource);
  setText("#total-revenue", money(overview.total_revenue));
  setText("#customer-count", overview.customer_count.toLocaleString());
  setText("#purchase-rate", `${overview.purchase_rate}%`);
  setText("#average-rating", `${overview.average_rating} / 5`);

  const previousSelection = customerSelect.value;
  customerSelect.replaceChildren();
  data.customers.forEach((customer) => {
    const option = document.createElement("option");
    option.value = customer.customer_id;
    option.textContent = customer.customer_id;
    customerSelect.append(option);
  });
  const desiredSelection = data.customers.some((item) => item.customer_id === selected.customer_id)
    ? selected.customer_id
    : previousSelection;
  customerSelect.value = desiredSelection || selected.customer_id;

  setText("#customer-name", selected.customer_id);
  setText("#customer-avatar", selected.customer_id.slice(0, 1).toUpperCase());
  setText("#segment-label", selected.segment);
  document.querySelector("#segment-label").className = `segment-pill ${segmentClass(selected.segment)}`;
  setText("#predicted-spending", money(selected.predicted_spending));
  setText("#purchase-probability", `${selected.purchase_probability}%`);
  document.querySelector("#probability-bar").style.width = `${selected.purchase_probability}%`;
  renderRecommendations(selected.recommendations);
  renderCategories(data.categories);
  renderSalesTrend(data.sales_trend);
  renderCustomers(data.customers, selected.customer_id);

  setText(
    "#spending-mae",
    metrics.spending_cv_mae == null ? "Not available" : `${money(metrics.spending_cv_mae)}`
  );
  setText(
    "#purchase-accuracy",
    metrics.purchase_cv_accuracy == null ? "Not available" : `${metrics.purchase_cv_accuracy}%`
  );
}

async function loadAnalysis(customerId) {
  const thisRequest = ++requestNumber;
  uploadButton.disabled = true;
  uploadButton.textContent = "Analyzing…";
  hideStatus();
  try {
    let response;
    if (currentFile) {
      const body = new FormData();
      body.append("file", currentFile);
      if (customerId) body.append("customer_id", customerId);
      response = await fetch(apiUrl("/api/analysis"), { method: "POST", body });
    } else {
      const query = customerId ? `?customer_id=${encodeURIComponent(customerId)}` : "";
      response = await fetch(apiUrl(`/api/analysis${query}`));
    }
    const result = await response.json();
    if (thisRequest !== requestNumber) return;
    if (!response.ok) throw new Error(result.error || "Unable to analyze this dataset.");
    renderDashboard(result);
  } catch (error) {
    if (thisRequest === requestNumber) {
      const message =
        error instanceof TypeError
          ? `Cannot reach the Flask API at ${apiUrl("/api/analysis")}. Start it with "python app.py".`
          : error.message;
      showStatus(message, "error");
    }
  } finally {
    if (thisRequest === requestNumber) {
      uploadButton.disabled = false;
      uploadButton.innerHTML = 'Analyze data <span aria-hidden="true">→</span>';
    }
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  document.querySelector("#file-name").textContent = file ? file.name : "";
});

uploadForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    showStatus("Choose a CSV file first, or download the sample CSV to get started.", "error");
    return;
  }
  currentFile = file;
  currentSource = `Uploaded: ${file.name}`;
  loadAnalysis();
});

customerSelect.addEventListener("change", () => loadAnalysis(customerSelect.value));

loadAnalysis();
