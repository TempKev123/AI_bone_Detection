import sys
import os
import torch
from torch import optim
from torchvision import datasets, transforms, models
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QComboBox, QMeskosageBox, QProgressBar, QVBoxLayout, QApplication, QScrollArea, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PIL import Image


class RetrainThread(QThread):
    update_progress = pyqtSignal(int)
    training_done = pyqtSignal(str)

    def __init__(self, model, image_dir):
        super().__init__()
        self.model = model
        self.image_dir = image_dir

    def run(self):
        # Image augmentation
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Dataset loading
        dataset = datasets.ImageFolder(self.image_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Model training
        self.model.train()
        num_epochs = 2
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Update progress
                progress = int(100 * (epoch * len(dataloader) + i + 1) / (num_epochs * len(dataloader)))
                self.update_progress.emit(progress)

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

        # Save the model
        torch.save(self.model.state_dict(), 'retrained_model.pth')
        self.training_done.emit("The model has been retrained and saved!")


class XRayApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = self.load_model()
        self.image_dir = 'saved_images/'
        os.makedirs(self.image_dir, exist_ok=True)

    def initUI(self):
        self.setWindowTitle('X-ray Fracture Detection and Training System')
        self.setGeometry(100, 100, 600, 500)

        self.layout = QVBoxLayout(self)

        # Image display area
        self.label = QLabel('Image Display Area', self)
        self.label.setFixedSize(300, 200)
        self.label.setStyleSheet("border: 1px solid black;")
        self.layout.addWidget(self.label)

        # Upload button
        self.btnLoad = QPushButton('Upload X-ray Images', self)
        self.btnLoad.clicked.connect(self.load_images)
        self.layout.addWidget(self.btnLoad)

        # Label for fracture or no fracture
        self.comboLabel = QComboBox(self)
        self.comboLabel.addItems(['No Fracture', 'Fracture'])
        self.layout.addWidget(self.comboLabel)

        # Save and Label button
        self.btnSave = QPushButton('Save and Label', self)
        self.btnSave.clicked.connect(self.save_image)
        self.layout.addWidget(self.btnSave)

        # Auto detect fracture button
        self.btnPredict = QPushButton('Auto Detect Fracture', self)
        self.btnPredict.clicked.connect(self.predict)
        self.layout.addWidget(self.btnPredict)

        # Retrain model button
        self.btnRetrain = QPushButton('Incremental Model Training', self)
        self.btnRetrain.clicked.connect(self.retrain_model)
        self.layout.addWidget(self.btnRetrain)

        # Progress bar
        self.progressBar = QProgressBar(self)
        self.layout.addWidget(self.progressBar)

        # Scroll area to display results
        self.scroll_area = QScrollArea(self)
        self.result_frame = QFrame(self)
        self.result_layout = QVBoxLayout(self.result_frame)
        self.result_frame.setLayout(self.result_layout)
        self.scroll_area.setWidget(self.result_frame)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

    def load_model(self):
        model = models.resnet18(weights=None)  # Do not load pretrained weights
        model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Set for binary classification
        model_path = r"E:\360\resnet18_model.pth"  # Model path

        try:
            checkpoint = torch.load(model_path)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model file not found!")
        except RuntimeError as e:
            print(f"Error loading model: {e}")
        model.eval()
        return model

    def load_images(self):
        options = QFileDialog.Options()
        # Allow selection of multiple files
        fileNames, _ = QFileDialog.getOpenFileNames(self, 'Select X-ray Images', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)', options=options)
        if fileNames:
            self.images = []
            self.image_paths = fileNames  # Store paths for later
            
            # Manually remove existing widgets from the result layout
            for i in reversed(range(self.result_layout.count())):
                widget = self.result_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

            # Load and display images
            for fileName in fileNames:
                try:
                    image = Image.open(fileName).convert("RGB")
                    self.images.append(image)

                    # Display the first image in the QLabel
                    pixmap = QPixmap(fileName)
                    self.label.setPixmap(pixmap.scaled(self.label.size()))

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def save_image(self):
        if hasattr(self, 'image'):
            label = self.comboLabel.currentText()
            image_name = os.path.basename(self.image_path)
            save_path = os.path.join(self.image_dir, label + '_' + image_name)
            self.image.save(save_path)
            QMessageBox.information(self, 'Success', f'Image saved as {save_path}')

    def predict(self):
        for image_path in self.image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                preprocess = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    output = self.model(img_tensor)
                    prediction = torch.argmax(output, 1).item()

                result = 'Fracture' if prediction == 1 else 'No Fracture'
                result_label = QLabel(f"{os.path.basename(image_path)}: {result}", self)
                self.result_layout.addWidget(result_label)
            except Exception as e:
                result_label = QLabel(f"Failed to process {os.path.basename(image_path)}: {str(e)}", self)
                self.result_layout.addWidget(result_label)

    def retrain_model(self):
        self.thread = RetrainThread(self.model, self.image_dir)
        self.thread.update_progress.connect(self.progressBar.setValue)
        self.thread.training_done.connect(lambda msg: QMessageBox.information(self, 'Success', msg))
        self.thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = XRayApp()
    ex.show()
    sys.exit(app.exec_())
import sys
import os
import torch
from torch import optim
from torchvision import datasets, transforms, models
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox, QProgressBar, QVBoxLayout, QApplication, QScrollArea, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PIL import Image


class RetrainThread(QThread):
    update_progress = pyqtSignal(int)
    training_done = pyqtSignal(str)

    def __init__(self, model, image_dir):
        super().__init__()
        self.model = model
        self.image_dir = image_dir

    def run(self):
        # Image augmentation
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Dataset loading
        dataset = datasets.ImageFolder(self.image_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Model training
        self.model.train()
        num_epochs = 2
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Update progress
                progress = int(100 * (epoch * len(dataloader) + i + 1) / (num_epochs * len(dataloader)))
                self.update_progress.emit(progress)

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

        # Save the model
        torch.save(self.model.state_dict(), 'retrained_model.pth')
        self.training_done.emit("The model has been retrained and saved!")


class XRayApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = self.load_model()
        self.image_dir = 'saved_images/'
        os.makedirs(self.image_dir, exist_ok=True)

    def initUI(self):
        self.setWindowTitle('X-ray Fracture Detection and Training System')
        self.setGeometry(100, 100, 600, 500)

        self.layout = QVBoxLayout(self)

        # Image display area
        self.label = QLabel('Image Display Area', self)
        self.label.setFixedSize(300, 200)
        self.label.setStyleSheet("border: 1px solid black;")
        self.layout.addWidget(self.label)

        # Upload button
        self.btnLoad = QPushButton('Upload X-ray Images', self)
        self.btnLoad.clicked.connect(self.load_images)
        self.layout.addWidget(self.btnLoad)

        # Label for fracture or no fracture
        self.comboLabel = QComboBox(self)
        self.comboLabel.addItems(['No Fracture', 'Fracture'])
        self.layout.addWidget(self.comboLabel)

        # Save and Label button
        self.btnSave = QPushButton('Save and Label', self)
        self.btnSave.clicked.connect(self.save_image)
        self.layout.addWidget(self.btnSave)

        # Auto detect fracture button
        self.btnPredict = QPushButton('Auto Detect Fracture', self)
        self.btnPredict.clicked.connect(self.predict)
        self.layout.addWidget(self.btnPredict)

        # Retrain model button
        self.btnRetrain = QPushButton('Incremental Model Training', self)
        self.btnRetrain.clicked.connect(self.retrain_model)
        self.layout.addWidget(self.btnRetrain)

        # Progress bar
        self.progressBar = QProgressBar(self)
        self.layout.addWidget(self.progressBar)

        # Scroll area to display results
        self.scroll_area = QScrollArea(self)
        self.result_frame = QFrame(self)
        self.result_layout = QVBoxLayout(self.result_frame)
        self.result_frame.setLayout(self.result_layout)
        self.scroll_area.setWidget(self.result_frame)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

    def load_model(self):
        model = models.resnet18(weights=None)  # Do not load pretrained weights
        model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Set for binary classification
        model_path = r"E:\360\resnet18_model.pth"  # Model path

        try:
            checkpoint = torch.load(model_path)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model file not found!")
        except RuntimeError as e:
            print(f"Error loading model: {e}")
        model.eval()
        return model

    def load_images(self):
        options = QFileDialog.Options()
        # Allow selection of multiple files
        fileNames, _ = QFileDialog.getOpenFileNames(self, 'Select X-ray Images', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)', options=options)
        if fileNames:
            self.images = []
            self.image_paths = fileNames  # Store paths for later
            
            # Manually remove existing widgets from the result layout
            for i in reversed(range(self.result_layout.count())):
                widget = self.result_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

            # Load and display images
            for fileName in fileNames:
                try:
                    image = Image.open(fileName).convert("RGB")
                    self.images.append(image)

                    # Display the first image in the QLabel
                    pixmap = QPixmap(fileName)
                    self.label.setPixmap(pixmap.scaled(self.label.size()))

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def save_image(self):
        if hasattr(self, 'image'):
            label = self.comboLabel.currentText()
            image_name = os.path.basename(self.image_path)
            save_path = os.path.join(self.image_dir, label + '_' + image_name)
            self.image.save(save_path)
            QMessageBox.information(self, 'Success', f'Image saved as {save_path}')

    def predict(self):
        for image_path in self.image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                preprocess = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    output = self.model(img_tensor)
                    prediction = torch.argmax(output, 1).item()

                result = 'Fracture' if prediction == 1 else 'No Fracture'
                result_label = QLabel(f"{os.path.basename(image_path)}: {result}", self)
                self.result_layout.addWidget(result_label)
            except Exception as e:
                result_label = QLabel(f"Failed to process {os.path.basename(image_path)}: {str(e)}", self)
                self.result_layout.addWidget(result_label)

    def retrain_model(self):
        self.thread = RetrainThread(self.model, self.image_dir)
        self.thread.update_progress.connect(self.progressBar.setValue)
        self.thread.training_done.connect(lambda msg: QMessageBox.information(self, 'Success', msg))
        self.thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = XRayApp()
    ex.show()
    sys.exit(app.exec_())
import sys
import os
import torch
from torch import optim
from torchvision import datasets, transforms, models
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox, QProgressBar, QVBoxLayout, QApplication, QScrollArea, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PIL import Image


class RetrainThread(QThread):
    update_progress = pyqtSignal(int)
    training_done = pyqtSignal(str)

    def __init__(self, model, image_dir):
        super().__init__()
        self.model = model
        self.image_dir = image_dir

    def run(self):
        # Image augmentation
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Dataset loading
        dataset = datasets.ImageFolder(self.image_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Model training
        self.model.train()
        num_epochs = 2
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Update progress
                progress = int(100 * (epoch * len(dataloader) + i + 1) / (num_epochs * len(dataloader)))
                self.update_progress.emit(progress)

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

        # Save the model
        torch.save(self.model.state_dict(), 'retrained_model.pth')
        self.training_done.emit("The model has been retrained and saved!")


class XRayApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = self.load_model()
        self.image_dir = 'saved_images/'
        os.makedirs(self.image_dir, exist_ok=True)

    def initUI(self):
        self.setWindowTitle('X-ray Fracture Detection and Training System')
        self.setGeometry(100, 100, 600, 500)

        self.layout = QVBoxLayout(self)

        # Image display area
        self.label = QLabel('Image Display Area', self)
        self.label.setFixedSize(300, 200)
        self.label.setStyleSheet("border: 1px solid black;")
        self.layout.addWidget(self.label)

        # Upload button
        self.btnLoad = QPushButton('Upload X-ray Images', self)
        self.btnLoad.clicked.connect(self.load_images)
        self.layout.addWidget(self.btnLoad)

        # Label for fracture or no fracture
        self.comboLabel = QComboBox(self)
        self.comboLabel.addItems(['No Fracture', 'Fracture'])
        self.layout.addWidget(self.comboLabel)

        # Save and Label button
        self.btnSave = QPushButton('Save and Label', self)
        self.btnSave.clicked.connect(self.save_image)
        self.layout.addWidget(self.btnSave)

        # Auto detect fracture button
        self.btnPredict = QPushButton('Auto Detect Fracture', self)
        self.btnPredict.clicked.connect(self.predict)
        self.layout.addWidget(self.btnPredict)

        # Retrain model button
        self.btnRetrain = QPushButton('Incremental Model Training', self)
        self.btnRetrain.clicked.connect(self.retrain_model)
        self.layout.addWidget(self.btnRetrain)

        # Progress bar
        self.progressBar = QProgressBar(self)
        self.layout.addWidget(self.progressBar)

        # Scroll area to display results
        self.scroll_area = QScrollArea(self)
        self.result_frame = QFrame(self)
        self.result_layout = QVBoxLayout(self.result_frame)
        self.result_frame.setLayout(self.result_layout)
        self.scroll_area.setWidget(self.result_frame)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

    def load_model(self):
        model = models.resnet18(weights=None)  # Do not load pretrained weights
        model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Set for binary classification
        model_path = r"E:\360\resnet18_model.pth"  # Model path

        try:
            checkpoint = torch.load(model_path)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model file not found!")
        except RuntimeError as e:
            print(f"Error loading model: {e}")
        model.eval()
        return model

    def load_images(self):
        options = QFileDialog.Options()
        # Allow selection of multiple files
        fileNames, _ = QFileDialog.getOpenFileNames(self, 'Select X-ray Images', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)', options=options)
        if fileNames:
            self.images = []
            self.image_paths = fileNames  # Store paths for later
            
            # Manually remove existing widgets from the result layout
            for i in reversed(range(self.result_layout.count())):
                widget = self.result_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

            # Load and display images
            for fileName in fileNames:
                try:
                    image = Image.open(fileName).convert("RGB")
                    self.images.append(image)

                    # Display the first image in the QLabel
                    pixmap = QPixmap(fileName)
                    self.label.setPixmap(pixmap.scaled(self.label.size()))

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def save_image(self):
        if hasattr(self, 'image'):
            label = self.comboLabel.currentText()
            image_name = os.path.basename(self.image_path)
            save_path = os.path.join(self.image_dir, label + '_' + image_name)
            self.image.save(save_path)
            QMessageBox.information(self, 'Success', f'Image saved as {save_path}')

    def predict(self):
        for image_path in self.image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                preprocess = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    output = self.model(img_tensor)
                    prediction = torch.argmax(output, 1).item()

                result = 'Fracture' if prediction == 1 else 'No Fracture'
                result_label = QLabel(f"{os.path.basename(image_path)}: {result}", self)
                self.result_layout.addWidget(result_label)
            except Exception as e:
                result_label = QLabel(f"Failed to process {os.path.basename(image_path)}: {str(e)}", self)
                self.result_layout.addWidget(result_label)

    def retrain_model(self):
        self.thread = RetrainThread(self.model, self.image_dir)
        self.thread.update_progress.connect(self.progressBar.setValue)
        self.thread.training_done.connect(lambda msg: QMessageBox.information(self, 'Success', msg))
        self.thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = XRayApp()
    ex.show()
    sys.exit(app.exec_())
import sys
import os
import torch
from torch import optim
from torchvision import datasets, transforms, models
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox, QProgressBar, QVBoxLayout, QApplication, QScrollArea, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PIL import Image


class RetrainThread(QThread):
    update_progress = pyqtSignal(int)
    training_done = pyqtSignal(str)

    def __init__(self, model, image_dir):
        super().__init__()
        self.model = model
        self.image_dir = image_dir

    def run(self):
        # Image augmentation
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Dataset loading
        dataset = datasets.ImageFolder(self.image_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Model training
        self.model.train()
        num_epochs = 2
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Update progress
                progress = int(100 * (epoch * len(dataloader) + i + 1) / (num_epochs * len(dataloader)))
                self.update_progress.emit(progress)

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

        # Save the model
        torch.save(self.model.state_dict(), 'retrained_model.pth')
        self.training_done.emit("The model has been retrained and saved!")


class XRayApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = self.load_model()
        self.image_dir = 'saved_images/'
        os.makedirs(self.image_dir, exist_ok=True)

    def initUI(self):
        self.setWindowTitle('X-ray Fracture Detection and Training System')
        self.setGeometry(100, 100, 600, 500)

        self.layout = QVBoxLayout(self)

        # Image display area
        self.label = QLabel('Image Display Area', self)
        self.label.setFixedSize(300, 200)
        self.label.setStyleSheet("border: 1px solid black;")
        self.layout.addWidget(self.label)

        # Upload button
        self.btnLoad = QPushButton('Upload X-ray Images', self)
        self.btnLoad.clicked.connect(self.load_images)
        self.layout.addWidget(self.btnLoad)

        # Label for fracture or no fracture
        self.comboLabel = QComboBox(self)
        self.comboLabel.addItems(['No Fracture', 'Fracture'])
        self.layout.addWidget(self.comboLabel)

        # Save and Label button
        self.btnSave = QPushButton('Save and Label', self)
        self.btnSave.clicked.connect(self.save_image)
        self.layout.addWidget(self.btnSave)

        # Auto detect fracture button
        self.btnPredict = QPushButton('Auto Detect Fracture', self)
        self.btnPredict.clicked.connect(self.predict)
        self.layout.addWidget(self.btnPredict)

        # Retrain model button
        self.btnRetrain = QPushButton('Incremental Model Training', self)
        self.btnRetrain.clicked.connect(self.retrain_model)
        self.layout.addWidget(self.btnRetrain)

        # Progress bar
        self.progressBar = QProgressBar(self)
        self.layout.addWidget(self.progressBar)

        # Scroll area to display results
        self.scroll_area = QScrollArea(self)
        self.result_frame = QFrame(self)
        self.result_layout = QVBoxLayout(self.result_frame)
        self.result_frame.setLayout(self.result_layout)
        self.scroll_area.setWidget(self.result_frame)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

    def load_model(self):
        model = models.resnet18(weights=None)  # Do not load pretrained weights
        model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Set for binary classification
        model_path = r"E:\360\resnet18_model.pth"  # Model path

        try:
            checkpoint = torch.load(model_path)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model file not found!")
        except RuntimeError as e:
            print(f"Error loading model: {e}")
        model.eval()
        return model

    def load_images(self):
        options = QFileDialog.Options()
        # Allow selection of multiple files
        fileNames, _ = QFileDialog.getOpenFileNames(self, 'Select X-ray Images', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)', options=options)
        if fileNames:
            self.images = []
            self.image_paths = fileNames  # Store paths for later
            
            # Manually remove existing widgets from the result layout
            for i in reversed(range(self.result_layout.count())):
                widget = self.result_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

            # Load and display images
            for fileName in fileNames:
                try:
                    image = Image.open(fileName).convert("RGB")
                    self.images.append(image)

                    # Display the first image in the QLabel
                    pixmap = QPixmap(fileName)
                    self.label.setPixmap(pixmap.scaled(self.label.size()))

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def save_image(self):
        if hasattr(self, 'image'):
            label = self.comboLabel.currentText()
            image_name = os.path.basename(self.image_path)
            save_path = os.path.join(self.image_dir, label + '_' + image_name)
            self.image.save(save_path)
            QMessageBox.information(self, 'Success', f'Image saved as {save_path}')

    def predict(self):
        for image_path in self.image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                preprocess = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    output = self.model(img_tensor)
                    prediction = torch.argmax(output, 1).item()

                result = 'Fracture' if prediction == 1 else 'No Fracture'
                result_label = QLabel(f"{os.path.basename(image_path)}: {result}", self)
                self.result_layout.addWidget(result_label)
            except Exception as e:
                result_label = QLabel(f"Failed to process {os.path.basename(image_path)}: {str(e)}", self)
                self.result_layout.addWidget(result_label)

    def retrain_model(self):
        self.thread = RetrainThread(self.model, self.image_dir)
        self.thread.update_progress.connect(self.progressBar.setValue)
        self.thread.training_done.connect(lambda msg: QMessageBox.information(self, 'Success', msg))
        self.thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = XRayApp()
    ex.show()
    sys.exit(app.exec_())
