import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

class ExpenditureTracker:
    """
    A class representing an Expenditure Tracker application.

    This class contains all the functions that allow the user to enter their expenditures in various categories,
    import/export the expenditure data to/from Excel sheets, and view a pie chart of the expenditures.
    """

    def __init__(self, root):
        """
        Initializes the ExpenditureTracker object.

        Input:
            root (tk.Tk): The root window of the application.

        Key Arguments and Logic:
            The function sets up the GUI components, including frames, input fields, buttons, and a pie chart.

        Output:
            None
        """
        self.root = root
        self.root.title("Expenditure Tracker")
        self.root.geometry("800x600")
        self.expenditures = {}  # Dictionary to store the expenditures

        # Create frames
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack()
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()
        self.chart_frame = tk.Frame(self.root)
        self.chart_frame.pack()

        # Create input widgets
        self.amount_label = tk.Label(self.input_frame, text="Amount:")
        self.amount_label.pack(side=tk.LEFT)
        self.amount_entry = tk.Entry(self.input_frame)  # Entry for amount
        self.amount_entry.pack(side=tk.LEFT)
        self.category_label = tk.Label(self.input_frame, text="Category:")
        self.category_label.pack(side=tk.LEFT)
        self.category_var = tk.StringVar()
        self.category_menu = ttk.Combobox(self.input_frame, textvariable=self.category_var)  # Category dropdown menu
        self.category_menu['values'] = ['Food', 'Transportation', 'Entertainment', 'Rent', 'Utilities', 'Other']
        self.category_menu.pack(side=tk.LEFT)
        self.other_label = tk.Label(self.input_frame, text="Other Category:")
        self.other_label.pack(side=tk.LEFT)
        self.other_entry = tk.Entry(self.input_frame)  # Entry for other category
        self.other_entry.pack(side=tk.LEFT)

        # Create buttons
        self.add_button = tk.Button(self.button_frame, text="Add Expenditure", command=self.add_expenditure)  # Button to add expenditure
        self.add_button.pack(side=tk.LEFT)
        self.view_button = tk.Button(self.button_frame, text="View Expenditures", command=self.view_expenditures)  # Button to view expenditures
        self.view_button.pack(side=tk.LEFT)
        self.export_button = tk.Button(self.button_frame, text="Export to Excel", command=self.export_to_excel)  # Button to export to Excel
        self.export_button.pack(side=tk.LEFT)
        self.import_button = tk.Button(self.button_frame, text="Import from Excel", command=self.import_from_excel)  # Button to import from Excel
        self.import_button.pack(side=tk.LEFT)

        # Create pie chart
        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.pie([100], labels=['No Data'], autopct='%1.1f%%')  # Initial pie chart
        self.ax.axis('equal')
        self.canvas = FigureCanvasTkAgg(self.figure, self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def add_expenditure(self):
        """
        Adds a new expenditure to the tracker.

        Input:
            None

        Key Arguments and Logic:
            The function gets the amount and category from the input fields, updates the expenditures dictionary,
            and then calls the view_expenditures method to update the pie chart.

        Output:
            None
        """
        amount = float(self.amount_entry.get())
        category = self.category_var.get()
        if category == 'Other':
            category = self.other_entry.get()
        if category in self.expenditures:
            self.expenditures[category] += amount
        else:
            self.expenditures[category] = amount
        self.amount_entry.delete(0, tk.END)
        self.other_entry.delete(0, tk.END)
        self.view_expenditures()

    def view_expenditures(self):
        """
        Updates the pie chart to display the current expenditures.

        Input:
            None

        Key Arguments and Logic:
            The function clears the current pie chart, gets the labels and sizes from the expenditures dictionary,
            and then draws a new pie chart.

        Output:
            None
        """
        self.ax.clear()
        labels = list(self.expenditures.keys())
        sizes = list(self.expenditures.values())
        self.ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        self.ax.axis('equal')
        self.canvas.draw()

    def export_to_excel(self):
        """
        Exports the expenditures to an Excel file.

        Input:
            None

        Key Arguments and Logic:
            The function opens a file dialog for the user to select a file path, creates a pandas DataFrame from the expenditures dictionary,
            and then writes the DataFrame to the selected Excel file.

        Output:
            None
        """
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx")
        df = pd.DataFrame(list(self.expenditures.items()), columns=['Category', 'Amount'])
        df.to_excel(file_path, index=False)

    def import_from_excel(self):
        """
        Imports expenditures from an Excel file.

        Input:
            None

        Key Arguments and Logic:
            The function opens a file dialog for the user to select a file path, reads the Excel file into a pandas DataFrame,
            and then updates the expenditures dictionary from the DataFrame.

        Output:
            None
        """
        file_path = filedialog.askopenfilename(defaultextension=".xlsx")
        df = pd.read_excel(file_path)
        self.expenditures = dict(zip(df['Category'], df['Amount']))
        self.view_expenditures()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExpenditureTracker(root)
    root.mainloop()