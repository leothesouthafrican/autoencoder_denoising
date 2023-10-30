import random
import datetime
from jinja2 import Environment, FileSystemLoader
import subprocess
import sys
from weasyprint import HTML
import os

# Random data generation
def random_date(start, end):
    return start + datetime.timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())),
    )

def create_random_invoice():
    return {
        "invoice_number": random.randint(1000, 9999),
        "date": random_date(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31)).strftime('%Y-%m-%d'),
        "due_date": random_date(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31)).strftime('%Y-%m-%d'),
        "customer_name": "Customer " + str(random.randint(1, 100)),
        "customer_address": "1234 Random St, Random City, RS",
        "items": [{"name": f"Item {i}", "quantity": random.randint(1, 10), "price": round(random.uniform(10, 1000), 2)} for i in range(random.randint(1, 10))],
        "total": round(random.uniform(100, 5000), 2)
    }

# Template rendering
env = Environment(loader=FileSystemLoader('path_to_your_templates_directory'))
template = env.get_template("your_template_name.html")  # Make sure this matches your HTML file name

# Generate a PDF from the template
def render_pdf(template, context):
    rendered_html = template.render(context)
    HTML(string=rendered_html).write_pdf("output_invoice.pdf")

# Example usage
invoice_data = create_random_invoice()
render_pdf(template, invoice_data)
