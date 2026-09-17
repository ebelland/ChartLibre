import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QLineEdit, 
                             QCheckBox, QSlider)
from PySide6.QtCore import Qt

class GlassWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(500, 450)
        
        # Il titolo viene impostato qui e apparirà sulla barra nativa di macOS
        self.setWindowTitle("Impostazioni di Sistema")

        # 1. TRUCCO MACOS: Mantiene la barra del titolo standard ma rimuove lo sfondo opaco.
        # Questo permette all'effetto vetro di estendersi anche dietro la barra nativa.
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # 2. Configura il widget centrale trasparente
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Layout principale
        main_layout = QVBoxLayout(self.central_widget)
        # Margine superiore ridotto perché ora usiamo la barra del titolo nativa di macOS
        main_layout.setContentsMargins(30, 20, 30, 30) 
        main_layout.setSpacing(20)

        # --- TITOLO INTERNO (Sotto la barra nativa) ---
        title_label = QLabel("Pannello di Controllo", self.central_widget)
        title_label.setObjectName("TitleLabel")
        main_layout.addWidget(title_label)

        # --- CAMPO DI TESTO ---
        self.input_field = QLineEdit(self.central_widget)
        self.input_field.setPlaceholderText("Scrivi qualcosa qui...")
        main_layout.addWidget(self.input_field)

        # --- OPZIONI (CHECKBOX) ---
        checkbox_layout = QHBoxLayout()
        self.check_option1 = QCheckBox("Attiva Notifiche", self.central_widget)
        self.check_option2 = QCheckBox("Modalità Avanzata", self.central_widget)
        checkbox_layout.addWidget(self.check_option1)
        checkbox_layout.addWidget(self.check_option2)
        main_layout.addLayout(checkbox_layout)

        # --- CONTROLLO VOLUME (SLIDER) ---
        slider_layout = QVBoxLayout()
        slider_label = QLabel("Intensità:", self.central_widget)
        self.slider = QSlider(Qt.Orientation.Horizontal, self.central_widget)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.slider)
        main_layout.addLayout(slider_layout)

        # --- PULSANTI IN BASSO ---
        button_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Annulla", self.central_widget)
        self.btn_save = QPushButton("Salva Modifiche", self.central_widget)
        self.btn_save.setObjectName("PrimaryButton")
        
        button_layout.addWidget(self.btn_cancel)
        button_layout.addWidget(self.btn_save)
        main_layout.addLayout(button_layout)

        # QSS per Vetro Chiaro
        self.setStyleSheet("""
            QMainWindow {
                background: transparent;
            }
            QWidget {
                color: #1C1C1E;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
                font-size: 13px;
                background: transparent;
            }
            TitleLabel {
                font-size: 22px;
                font-weight: bold;
                color: #1C1C1E;
                margin-top: 10px;
            }
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.06);
                border: 1px solid rgba(0, 0, 0, 0.08);
                border-radius: 6px;
                padding: 8px 12px;
                color: #1C1C1E;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #007AFF;
                background-color: rgba(0, 0, 0, 0.02);
            }
            QPushButton {
                background-color: rgba(0, 0, 0, 0.05);
                border: 1px solid rgba(0, 0, 0, 0.05);
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                color: #1C1C1E;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.1);
            }
            PrimaryButton {
                background-color: #007AFF;
                border: none;
                color: white;
            }
            PrimaryButton:hover {
                background-color: #268FFF;
            }
        """)

    def showEvent(self, event):
        super().showEvent(event)
        # Forza lo sfondo pass-through a rendering avviato
        self.centralWidget().setStyleSheet("background: transparent;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Forza lo stile grafico nativo del Mac per integrare i pulsanti semaforo di sistema
    app.setStyle("macintosh")
    
    window = GlassWindow()
    window.show()
    sys.exit(app.exec())
