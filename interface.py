from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QMessageBox, QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import QTimer
import sys

class ParameterDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Parameters")
        layout = QFormLayout()

        self.n_input = QLineEdit()
        self.iter_input = QLineEdit()
        layout.addRow("N (board size):", self.n_input)
        layout.addRow("Max iterations:", self.iter_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_values(self):
        return self.n_input.text(), self.iter_input.text()


class TabuSearchInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tabu Search N-Queens")

        layout = QVBoxLayout()
        self.run_button = QPushButton("Run Tabu Search")
        self.run_button.clicked.connect(self.ask_parameters)
        layout.addWidget(self.run_button)

        # Visualization placeholder
        self.visual_label = QLabel("Visualization Area")
        self.visual_label.setStyleSheet("background-color: lightgray; min-height: 150px;")
        layout.addWidget(self.visual_label)

        # Current Solution matrix
        self.solution_matrix = QTableWidget(0, 0)
        layout.addWidget(QLabel("Current Solution"))
        layout.addWidget(self.solution_matrix)

        self.setLayout(layout)

        # Variables for search
        self.n = 0
        self.max_iters = 0
        self.iteration = 0

    def ask_parameters(self):
        dialog = ParameterDialog()
        if dialog.exec():
            n_str, iters_str = dialog.get_values()
            try:
                self.n = int(n_str)
                self.max_iters = int(iters_str)
                if self.n <= 0 or self.max_iters <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Error", "Enter valid positive integers.")
                return

            # Resize solution matrix
            self.solution_matrix.setRowCount(self.n)
            self.solution_matrix.setColumnCount(self.n)
            for i in range(self.n):
                for j in range(self.n):
                    self.solution_matrix.setItem(i, j, QTableWidgetItem(""))

            # Start tabu search (animated with timer)
            self.iteration = 0
            self.timer = QTimer()
            self.timer.timeout.connect(self.run_step)
            self.timer.start(500)  # update every 500ms

    def run_step(self):
        self.iteration += 1

        self.visual_label.setText(f"Iteration {self.iteration}/{self.max_iters}")

        # Example: Fill diagonal cells with "Q"
        if self.iteration <= self.n:
            self.solution_matrix.setItem(self.iteration-1, self.iteration-1, QTableWidgetItem("Q"))

        # If "solution found"
        if self.iteration == self.n:
            self.timer.stop()
            QMessageBox.information(self, "Solution Found",
                                    f"Solution found at iteration {self.iteration}!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TabuSearchInterface()
    window.show()
    sys.exit(app.exec())
