# import tkinter as tk
# from tkinter import ttk
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib.pyplot as plt
# import calendar

# class BillGeneratorApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title('Bill Generator App')
#         self.team_members = {}

#         # Create frames
#         self.main_frame = tk.Frame(self.root)
#         self.main_frame.pack(fill='both', expand=True)
#         self.top_frame = tk.Frame(self.main_frame)
#         self.top_frame.pack(fill='x')
#         self.middle_frame = tk.Frame(self.main_frame)
#         self.middle_frame.pack(fill='both', expand=True)
#         self.bottom_frame = tk.Frame(self.main_frame)
#         self.bottom_frame.pack(fill='x')

#         # Create labels and entries
#         self.name_label = tk.Label(self.top_frame, text='Name:')
#         self.name_label.pack(side='left')
#         self.name_entry = tk.Entry(self.top_frame)
#         self.name_entry.pack(side='left')
#         self.amount_label = tk.Label(self.top_frame, text='Amount:')
#         self.amount_label.pack(side='left')
#         self.amount_entry = tk.Entry(self.top_frame)
#         self.amount_entry.pack(side='left')
#         self.service_label = tk.Label(self.top_frame, text='Service:')
#         self.service_label.pack(side='left')
#         self.service_entry = tk.Entry(self.top_frame)
#         self.service_entry.pack(side='left')

#         # Create month dropdown
#         self.months = list(calendar.month_name)[1:]
#         self.month_var = tk.StringVar()
#         self.month_var.set(self.months[0])
#         self.month_menu = tk.OptionMenu(self.top_frame, self.month_var, *self.months)
#         self.month_menu.pack(side='left')

#         # Create buttons
#         self.add_button = tk.Button(self.top_frame, text='Add', command=self.add_team_member)
#         self.add_button.pack(side='left')
#         self.generate_button = tk.Button(self.top_frame, text='Generate Bill', command=self.generate_bill)
#         self.generate_button.pack(side='left')

#         # Create treeview
#         self.treeview = ttk.Treeview(self.middle_frame, columns=('Name', 'Amount', 'Service'), show='headings')
#         self.treeview.heading('Name', text='Name')
#         self.treeview.heading('Amount', text='Amount')
#         self.treeview.heading('Service', text='Service')
#         self.treeview.pack(fill='both', expand=True)

#         # Create figure and axis
#         self.figure = Figure(figsize=(8, 6), dpi=100)
#         self.ax = self.figure.add_subplot(111)
#         self.canvas = FigureCanvasTkAgg(self.figure, self.bottom_frame)
#         self.canvas.draw()
#         self.canvas.get_tk_widget().pack(fill='both', expand=True)
#         self.annot = None
#         self.annot_text = None
#         self.figure.canvas.mpl_connect("motion_notify_event", self.hover)

#     def add_team_member(self):
#         name = self.name_entry.get()
#         amount = float(self.amount_entry.get())
#         service = self.service_entry.get()
#         self.team_members[name] = {'Amount': amount, 'Service': service}
#         self.treeview.insert('', 'end', values=(name, amount, service))
#         self.name_entry.delete(0, 'end')
#         self.amount_entry.delete(0, 'end')
#         self.service_entry.delete(0, 'end')
#         self.update_bar_chart()

#     def update_bar_chart(self):
#         self.ax.clear()
#         names = list(self.team_members.keys())
#         amounts = [member['Amount'] for member in self.team_members.values()]
#         self.ax.bar(names, amounts)
#         self.ax.set_xlabel('Name')
#         self.ax.set_ylabel('Amount')
#         self.ax.set_title('Amount Owed by Each Member')
#         self.canvas.draw()

#     def generate_bill(self):
#         month = self.month_var.get()
#         bill = 'Bill for {}\n\n'.format(month)
#         for name, details in self.team_members.items():
#             bill += 'Name: {}\nAmount: {}\nService: {}\n\n'.format(name, details['Amount'], details['Service'])
#         print(bill)

#     def hover(self, event):
#         if event.inaxes == self.ax:
#             if self.annot is not None:
#                 self.annot.remove()
#                 self.annot = None
#             x = event.xdata
#             y = event.ydata
#             if x is not None and y is not None:
#                 for name, details in self.team_members.items():
#                     if abs(y - details['Amount']) < 1:
#                         self.annot = self.ax.annotate('Amount: {}'.format(details['Amount']), xy=(x, y), xytext=(5, 5), textcoords='offset points')
#                         self.canvas.draw_idle()

# if __name__ == '__main__':
#     root = tk.Tk()
#     app = BillGeneratorApp(root)
#     root.mainloop()


import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd

class BillGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Bill Generator App')
        self.team_members = {}

        # Create frames
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True)
        self.top_frame = tk.Frame(self.main_frame)
        self.top_frame.pack(fill='x')
        self.middle_frame = tk.Frame(self.main_frame)
        self.middle_frame.pack(fill='both', expand=True)
        self.bottom_frame = tk.Frame(self.main_frame)
        self.bottom_frame.pack(fill='x')

        # Create labels and entries
        self.name_label = tk.Label(self.top_frame, text='Name:')
        self.name_label.pack(side='left')
        self.name_entry = tk.Entry(self.top_frame)
        self.name_entry.pack(side='left')
        self.amount_label = tk.Label(self.top_frame, text='Amount:')
        self.amount_label.pack(side='left')
        self.amount_entry = tk.Entry(self.top_frame)
        self.amount_entry.pack(side='left')

        # Create service dropdown
        self.service_label = tk.Label(self.top_frame, text='Service:')
        self.service_label.pack(side='left')
        self.service_var = tk.StringVar()
        self.services = ['Phone', 'Phone+Insurance', 'Misc']
        self.service_menu = ttk.OptionMenu(self.top_frame, self.service_var, self.services[0], *self.services)
        self.service_menu.pack(side='left')

        # Create month dropdown
        self.month_label = tk.Label(self.top_frame, text='Month:')
        self.month_label.pack(side='left')
        self.month_var = tk.StringVar()
        self.months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        self.month_menu = ttk.OptionMenu(self.top_frame, self.month_var, self.months[0], *self.months)
        self.month_menu.pack(side='left')

        # Create buttons
        self.add_button = tk.Button(self.top_frame, text='Add', command=self.add_team_member)
        self.add_button.pack(side='left')
        self.generate_button = tk.Button(self.top_frame, text='Generate Bill', command=self.generate_bill)
        self.generate_button.pack(side='left')
        self.export_button = tk.Button(self.top_frame, text='Export to Excel', command=self.export_to_excel)
        self.export_button.pack(side='left')

        # Create treeview
        self.treeview = ttk.Treeview(self.middle_frame, columns=('Name', 'Amount', 'Service'), show='headings')
        self.treeview.heading('Name', text='Name')
        self.treeview.heading('Amount', text='Amount')
        self.treeview.heading('Service', text='Service')
        self.treeview.pack(fill='both', expand=True)

        # Create figure and axis
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.bottom_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.annotation = None

        # Create total label
        self.total_label = tk.Label(self.bottom_frame, text='Total: $0.00')
        self.total_label.pack(fill='x')

    def add_team_member(self):
        name = self.name_entry.get()
        amount = float(self.amount_entry.get())
        service = self.service_var.get()
        self.team_members[name] = {'Amount': amount, 'Service': service}
        self.treeview.insert('', 'end', values=(name, amount, service))
        self.name_entry.delete(0, 'end')
        self.amount_entry.delete(0, 'end')
        self.update_bar_chart()
        self.update_total()

    def update_bar_chart(self):
        self.ax.clear()
        names = list(self.team_members.keys())
        amounts = [member['Amount'] for member in self.team_members.values()]
        bars = self.ax.bar(names, amounts)
        self.ax.set_xlabel('Name')
        self.ax.set_ylabel('Amount')
        self.ax.set_title('Amount Owed by Each Member')
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')
        self.canvas.draw()

    def update_total(self):
        total = sum(member['Amount'] for member in self.team_members.values())
        self.total_label['text'] = f'Total: ${total:.2f}'

    def generate_bill(self):
        month = self.month_var.get()
        bill = f'Bill for {month}\n\n'
        for name, details in self.team_members.items():
            bill += f'Name: {name}\nAmount: {details["Amount"]}\nService: {details["Service"]}\n\n'
        print(bill)

    def export_to_excel(self):
        month = self.month_var.get()
        data = {'Name': [], 'Amount': [], 'Service': []}
        for name, details in self.team_members.items():
            data['Name'].append(name)
            data['Amount'].append(details['Amount'])
            data['Service'].append(details['Service'])
        df = pd.DataFrame(data)
        df.to_excel(f'{month}_bill.xlsx', index=False)

if __name__ == '__main__':
    root = tk.Tk()
    app = BillGeneratorApp(root)
    root.mainloop()