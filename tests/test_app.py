import io
import unittest
from pathlib import Path

from app import app


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_default_analysis_returns_predictions_and_segments(self):
        response = self.client.get("/api/analysis")

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertGreater(result["overview"]["customer_count"], 2)
        self.assertIn("predicted_spending", result["selected_customer"])
        self.assertIn(
            result["selected_customer"]["segment"],
            {"Premium", "Regular", "Occasional"},
        )
        self.assertIn("recommendations", result["selected_customer"])
        self.assertEqual(len(result["sales_trend"]), 5)

    def test_uploaded_csv_returns_selected_customer_analysis(self):
        sample_path = Path(__file__).resolve().parents[1] / "data" / "customers.csv"
        response = self.client.post(
            "/api/analysis",
            data={
                "file": (io.BytesIO(sample_path.read_bytes()), "customers.csv"),
                "customer_id": "C1008",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["selected_customer"]["customer_id"], "C1008")
        self.assertGreaterEqual(
            len(result["selected_customer"]["recommendations"]), 1
        )

    def test_unknown_customer_returns_clear_error(self):
        response = self.client.get("/api/analysis?customer_id=unknown")

        self.assertEqual(response.status_code, 400)
        self.assertIn("was not found", response.get_json()["error"])

    def test_upload_reports_missing_required_columns(self):
        response = self.client.post(
            "/api/analysis",
            data={"file": (io.BytesIO(b"customer_id,age\nC1,25\n"), "small.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("CSV columns do not match", response.get_json()["error"])
        self.assertIn("Found: customer_id, age", response.get_json()["error"])

    def test_upload_accepts_human_readable_headers_and_semicolon_delimiter(self):
        sample_path = Path(__file__).resolve().parents[1] / "data" / "customers.csv"
        sample = sample_path.read_text(encoding="utf-8-sig")
        lines = sample.splitlines()
        headers = [
            "Customer ID" if column == "customer_id" else column.replace("_", " ").title()
            for column in lines[0].split(",")
        ]
        csv_data = ";".join(headers) + "\n" + "\n".join(
            line.replace(",", ";") for line in lines[1:]
        )
        response = self.client.post(
            "/api/analysis",
            data={"file": (io.BytesIO(csv_data.encode()), "data.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["overview"]["customer_count"], 20)

    def test_upload_cleans_missing_values_and_duplicate_rows(self):
        csv_data = (
            "customer_id,age,gender,region,product_id,product_name,product_category,"
            "browsing_minutes,pages_viewed,rating,purchase_count,purchase_amount,purchase\n"
            "C1,25,Female,North,P1,Pack,Bags,10,5,,1,30,1\n"
            "C1,25,Female,North,P1,Pack,Bags,10,5,,1,30,1\n"
            "C1,25,Female,North,P2,Cup,Home,8,3,4,1,0,0\n"
            "C2,40,Male,South,P2,Cup,Home,15,7,3,4,0,0\n"
            "C2,40,Male,South,P3,Shoes,Footwear,20,9,5,4,80,1\n"
        )
        response = self.client.post(
            "/api/analysis",
            data={"file": (io.BytesIO(csv_data.encode()), "cleanup.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        overview = response.get_json()["overview"]
        self.assertEqual(overview["duplicate_rows_removed"], 1)
        self.assertEqual(overview["missing_values_cleaned"], 1)
        self.assertEqual(overview["interaction_count"], 4)

    def test_sample_csv_is_downloadable(self):
        response = self.client.get("/sample.csv")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"customer_id", response.data)
        response.close()

    def test_dashboard_renders_frontend(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Customer outlook", response.data)
        self.assertIn(b'href="/static/styles.css"', response.data)
        self.assertIn(b'src="/static/app.js"', response.data)

    def test_live_server_origin_can_call_api(self):
        response = self.client.get(
            "/api/analysis",
            headers={"Origin": "http://127.0.0.1:5500"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Access-Control-Allow-Origin"],
            "http://127.0.0.1:5500",
        )

    def test_unrelated_origin_is_not_allowed(self):
        response = self.client.get(
            "/api/analysis",
            headers={"Origin": "https://example.com"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)


if __name__ == "__main__":
    unittest.main()
