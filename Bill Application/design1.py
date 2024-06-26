import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt

class BillApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Team Bill Application")
        self.members = {}

        # Create frames
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(padx=10, pady=10)

        self.chart_frame = tk.Frame(self.root)
        self.chart_frame.pack(padx=10, pady=10)

        # Create input widgets
        self.name_label = tk.Label(self.input_frame, text="Name:")
        self.name_label.grid(row=0, column=0, padx=5, pady=5)

        self.name_entry = tk.Entry(self.input_frame, width=30)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)

        self.amount_label = tk.Label(self.input_frame, text="Amount:")
        self.amount_label.grid(row=1, column=0, padx=5, pady=5)

        self.amount_entry = tk.Entry(self.input_frame, width=30)
        self.amount_entry.grid(row=1, column=1, padx=5, pady=5)

        self.service_label = tk.Label(self.input_frame, text="Service:")
        self.service_label.grid(row=2, column=0, padx=5, pady=5)

        self.service_entry = tk.Entry(self.input_frame, width=30)
        self.service_entry.grid(row=2, column=1, padx=5, pady=5)

        self.date_range_label = tk.Label(self.input_frame, text="Date Range (e.g., 2022-01-01 to 2022-01-31):")
        self.date_range_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        self.date_range_entry = tk.Entry(self.input_frame, width=30)
        self.date_range_entry.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        self.add_button = tk.Button(self.input_frame, text="Add Member", command=self.add_member)
        self.add_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        self.chart_button = tk.Button(self.input_frame, text="Show Chart", command=self.show_chart)
        self.chart_button.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

        self.bill_button = tk.Button(self.input_frame, text="Generate Bill", command=self.generate_bill)
        self.bill_button.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

    def add_member(self):
        name = self.name_entry.get()
        amount = self.amount_entry.get()
        service = self.service_entry.get()

        if name and amount and service:
            self.members[name] = {"amount": float(amount), "service": service}
            self.name_entry.delete(0, tk.END)
            self.amount_entry.delete(0, tk.END)
            self.service_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Please fill in all fields.")

    def show_chart(self):
        if self.members:
            names = list(self.members.keys())
            amounts = [member["amount"] for member in self.members.values()]

            plt.bar(names, amounts)
            plt.xlabel("Name")
            plt.ylabel("Amount")
            plt.title("Amount Owed by Each Member")
            plt.show()
        else:
            messagebox.showerror("Error", "No members added.")

    def generate_bill(self):
        if self.members:
            date_range = self.date_range_entry.get()
            bill = f"Bill for {date_range}\n\n"

            for name, details in self.members.items():
                bill += f"{name}: {details['amount']} for {details['service']}\n"

            messagebox.showinfo("Bill", bill)
        else:
            messagebox.showerror("Error", "No members added.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BillApp(root)
    root.mainloop()