from PyQt5 import uic, QtWidgets, QtSerialPort, QtCore, QtPrintSupport, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
import qtmodern.styles
import qtmodern.windows
import numpy as np
import imutils
from datetime import datetime
import urllib.request
from pypylon_opencv_viewer import BaslerOpenCVViewer
from pypylon import pylon
import configparser
import timeit
from imutils import contours
from skimage import measure
from tinydb import TinyDB, Query, table, where
from tinydb.operations import delete
from INSPECTION_MODE import inspection_modes
import os, sys
import shutil
import time
import cv2
from numpy import random
import pathlib
import xml.etree.ElementTree as ET
import mysql.connector
import socket
import serial
import re
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
import uuid
import logging

def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                logging.warning("skip: %s" % option)
        except:
            logging.warning("exception on %s!" % option)
            dict1[option] = None
    return dict1


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def check_img_folder():
    try:
        if not os.path.exists(resource_path(f'Images_data/{datetime.today()}')):
            os.makedirs(resource_path(f'Images_data/{datetime.today().strftime("%m%d%Y")}'))
            os.makedirs(resource_path(f'Images_data/{datetime.today().strftime("%m%d%Y")}/OK'))
            os.makedirs(resource_path(f'Images_data/{datetime.today().strftime("%m%d%Y")}/NG'))
    except Exception as e:
        logging.warning(e)


class Thread_scanner(QThread):
    get_epass = pyqtSignal(str)

    def run(self):

        try:
            port = MainWindow.io_db.all()[0]["com_scanner"]

            try:
                serialPort = serial.Serial(port, baudrate=9600,
                                       bytesize=8, timeout=0.25, stopbits=serial.STOPBITS_ONE)
                while (1):
                    if (serialPort.in_waiting > 0):
                        serialString = serialPort.readline().decode()
                        self.get_epass.emit(serialString)
            except Exception as e:
                logging.warning(e)

        except Exception as e:
            logging.warning("Cannot open com for epass"+e)
            
class recheck_single(QThread):

    changePixmap = pyqtSignal(np.ndarray)
    change_isp_time = pyqtSignal(float)
    save_data = pyqtSignal(np.ndarray, bool, str)
    update_item_status = pyqtSignal(str, bool)
    update_f_res = pyqtSignal(bool)

    def run(self):
        try:
            res_dict = dict()
            starttime = timeit.default_timer()
            frame = MainWindow.viewer1.get_image()
            specs = list()
            root = MainWindow.inspection_sq
            for i,r in enumerate(root):
                if r.tag == MainWindow.recheck_iem:
                    specs = list()
                    for sub in r:
                        specs.append(sub.text)
            detected, res = inspection_modes.inspection_modes[MainWindow.recheck_iem]([frame] + specs)
            if detected is not None:
                self.changePixmap.emit(detected)
            self.update_item_status.emit(MainWindow.recheck_iem, res)
            res_dict[MainWindow.recheck_iem] = res
            self.save_data.emit(detected, res, MainWindow.recheck_iem)
            MainWindow.result_image[MainWindow.recheck_iem] = detected

            if False in list(res_dict.values()):
                self.update_f_res.emit(False)
            else:
                self.update_f_res.emit(True)

            isp_time = float(timeit.default_timer() - starttime)
            print(float(isp_time))
            self.change_isp_time.emit(float(str(isp_time)))

        except Exception as e:
            logging.warning(e)


class Thread(QThread):


    changePixmap = pyqtSignal(np.ndarray)
    change_isp_time = pyqtSignal(float)
    save_data = pyqtSignal(np.ndarray, bool, str)
    update_item_status = pyqtSignal(str, bool)
    update_f_res = pyqtSignal(bool)

    def run(self):
        try:
            res_dict = dict()
            starttime = timeit.default_timer()
            # frame = MainWindow.viewer1.get_image()
            frame = cv2.imread(r'D:\VISION INSPECTION SYSTEM\Images_data\3b5facaf-fd89-48d4-a1aa-cf8e1da54e03_CURV_LED_OFF.png')
            root = MainWindow.inspection_sq
            for i,r in enumerate(root):
                # Get SPEC from xml
                specs = list()
                for sub in r:
                    specs.append(sub.text)
                print(f"Start inspecting {r.tag}...")
                detected, res = inspection_modes.inspection_modes[r.tag]([frame] + specs)
                if detected is not None:
                    self.changePixmap.emit(detected)

                if type(res) is dict:
                    for re in res:
                        self.save_data.emit(detected, res[re], "_LED_"+re.upper())

                    if False in res.values():
                        res = False
                    else:
                        res = True


                    self.update_item_status.emit(r.tag, res)
                    res_dict[r.tag] = res

                else:
                    self.update_item_status.emit(r.tag, res)
                    res_dict[r.tag] = res
                    self.save_data.emit(detected, res, str(r.tag))
                MainWindow.result_image[str(r.tag)] = detected
                
            if False in list(res_dict.values()):
                self.update_f_res.emit(False)
            else:
                self.update_f_res.emit(True)

            isp_time = float(timeit.default_timer() - starttime)
            print(float(isp_time))
            self.change_isp_time.emit(float(str(isp_time)))

        except Exception as e:
            logging.warning(e)


class load_model(QThread):

    change_isp_time = pyqtSignal(str)

    def run(self):
        try:
            for i in MainWindow.net_path:
                net_pack = inspection_modes.load_tf_model(i[0], i[1])
                MainWindow.dl_net.append(net_pack[0])
                self.change_isp_time.emit(f"[DONE] Loaded model in {str(net_pack[2])[:7]} second(s)")

        except Exception as e:
            logging.warning(e)


class MainWindow(QtWidgets.QMainWindow):

    #Init settings db
    app_database = TinyDB('DATA/app_db.json')
    camera_db = app_database.table('camera_settings')
    sql_db = app_database.table('database_settings')
    io_db = app_database.table('io_settings')
    inspection_sq = None
    viewer1, viewer2 = None, None
    net_path = list()
    dl_net = list()
    result_image = dict()
    recheck_iem = None
    board_mapping = None
    IP = None

    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('ui/main.ui', self)
        #LOAD BACKEND
        self.mycursor = None
        self.mydb = None
        for i in MainWindow.camera_db.all():
            if True:
                info = None
                for x in pylon.TlFactory.GetInstance().EnumerateDevices():
                    if x.GetSerialNumber() == i['id']:
                        info = x
                        break
                else:
                    print('Camera with {} serial number not found'.format(i['id']))

                # VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
                if info is not None:
                    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
                    camera.Open()
                    if MainWindow.viewer1 is None:
                        MainWindow.viewer1 = BaslerOpenCVViewer(camera)
                        print(f'Camera 1 - serial number: {i["id"]}-OK')

                    else:
                        MainWindow.viewer2 = BaslerOpenCVViewer(camera)
                        print(f'Camera 2 - serial number: {i["id"]}-OK')

        try:
            #LOAD FRONTEND
            self.setWindowIcon(QtGui.QIcon(resource_path(r'ui/iconfinder_google_cardboard_5710528.png')))
            self.display_version = self.findChild(QtWidgets.QLabel, 'label_8')
            self.display_version.setText("version "+__version__)
            self.cb_sq = self.findChild(QtWidgets.QComboBox, 'comboBox')
            self.btn_control_sq = self.findChild(QtWidgets.QPushButton, 'pushButton')
            self.btn_rf_sq = self.findChild(QtWidgets.QToolButton, 'toolButton')
            self.btn_load_sq = self.findChild(QtWidgets.QPushButton, 'pushButton_3')
            self.btn_clear_log = self.findChild(QtWidgets.QPushButton, 'pushButton_4')
            self.btn_open_imgloc = self.findChild(QtWidgets.QPushButton, 'pushButton_5')
            self.btn_run = self.findChild(QtWidgets.QPushButton, 'pushButton_2')
            # self.btn_save_sq = self.findChild(QtWidgets.QPushButton, 'pushButton_5')
            # self.btn_create_sq = self.findChild(QtWidgets.QPushButton, 'pushButton_6')
            self.display_list_sq = self.findChild(QtWidgets.QListWidget, 'listWidget')
            self.line_new_sq = self.findChild(QtWidgets.QLineEdit, 'lineEdit')
            self.log_terminal = self.findChild(QtWidgets.QTextEdit, 'textEdit')
            self.display1 = self.findChild(QtWidgets.QLabel, 'label_6')
            self.display2 = self.findChild(QtWidgets.QLabel, 'label_12')
            self.display_isp_res = self.findChild(QtWidgets.QLabel, 'label_5')
            self.display_current_sq = self.findChild(QtWidgets.QLabel, 'label_2')

            self.final_res = list()

            self.lcdnumber = self.findChild(QtWidgets.QLCDNumber, 'lcdNumber')
            self.epass_input = self.findChild(QtWidgets.QLineEdit, 'lineEdit')
            self.btn_cs = self.findChild(QtWidgets.QAction, 'actionCamera_Settings_2')
            self.btn_oes = self.findChild(QtWidgets.QAction, 'actionFind_ODD_EVEN_specs')
            self.btn_wss = self.findChild(QtWidgets.QAction, 'actionFind_WHITE_SPOTS_LED_specs')
            self.btn_dcs = self.findChild(QtWidgets.QAction, 'actionFind_DIFF_COLOR_specs')
            self.btn_ts = self.findChild(QtWidgets.QAction, 'actionFind_LED_specs')
            self.btn_rv = self.findChild(QtWidgets.QAction, 'actionRESULT_VIEWER')
            self.cb_auto_change = self.findChild(QtWidgets.QCheckBox, 'checkBox')
            self.st_scanner = self.findChild(QtWidgets.QLabel, 'label_9')
            self.st_db = self.findChild(QtWidgets.QLabel, 'label_11')
            self.progress = self.findChild(QtWidgets.QProgressBar, 'progressBar')

            if len(MainWindow.camera_db.all()) == 1:
                self.display2.hide()

            # LOAD SIGNAL
            # self.btn_cs.triggered.connect(self.open_cs)
            self.btn_wss.triggered.connect(self.open_wss)
            self.btn_dcs.triggered.connect(self.open_dcs)
            self.btn_oes.triggered.connect(self.oes)
            self.btn_rv.triggered.connect(self.open_rv)
            self.btn_ts.triggered.connect(self.open_ts)
            self.btn_clear_log.clicked.connect(self.clear_log)
            self.btn_load_sq.clicked.connect(self.load_sq)
            self.btn_control_sq.clicked.connect(self.control_sq)
            self.btn_run.clicked.connect(self.run)
            self.cb_auto_change.toggled.connect(lambda: self.change_state())
            self.btn_rf_sq.clicked.connect(self.rf)
            self.btn_open_imgloc.clicked.connect(self.open_img_loc)
            self.display_list_sq.itemClicked.connect(self.show_item_info)
            self.display_list_sq.itemDoubleClicked.connect(self.recheck)

            # LOAD BACKEND
            self.init_db()
            self.machine_ip = self.get_ip()
            MainWindow.IP = self.machine_ip

            self.write_show_log(f'Machine IP: {self.machine_ip}')
            # self.btn_save_sq.clicked.connect(self.save_sq)
            # self.btn_create_sq.clicked.connect(self.create_sq)

            self.load_cb(self.cb_sq, self.get_sq_list())
            self.auto_change_mode = self.cb_auto_change.isChecked()
            self.mapping_data = self.get_mapping()
            self.write_show_log(f'Load mapping data from db: OK')
            self.update_space()
            self.get_board_mapping()

        except Exception as e:
            logging.warning(e)

        self.check_connection()

        try:
            self.thread_epass = Thread_scanner(self)
            self.thread_epass.get_epass.connect(self.set_epass)
            self.thread_epass.start()
        except Exception as e:
            logging.warning(e)

        self.timer_rf_mapping = QtCore.QTimer(self, interval=1000)
        self.timer_rf_mapping.timeout.connect(self.rf_mapping)
        self.timer_rf_mapping.timeout.connect(self.get_board_mapping)
        self.timer_rf_mapping.start()

        self.write_show_log(f'Auto change seq: {self.auto_change_mode}')
        self.show()

    def init_db(self):
        try:

            self.mydb = self.connect_sql()
            self.mycursor = self.mydb.cursor()
            if self.mydb:
                self.st_db.setStyleSheet("color: rgb(0, 0, 0);background-color: rgb(85, 255, 127);")
                self.st_db.setText("DATABASE OK")
        except:
            self.st_db.setStyleSheet("color: rgb(0, 0, 0);background-color: rgb(255, 0, 0);")
            self.st_db.setText("DATABASE FAILED")


    def open_img_loc(self):
        os.startfile(resource_path('Images_data'))


    def rf(self):
        self.load_cb(self.cb_sq, self.get_sq_list())

    @pyqtSlot(str)
    def set_epass(self, sn):
        try:
            text = re.sub("[^0-9a-zA-Z]", "", sn)
            self.epass_input.setText(text)
            self.run()

        except Exception as e:
            logging.warning(e)

    def update_space(self):
        try:
            total, used, free = shutil.disk_usage("/")

            used = used // (2 ** 30)
            total = total // (2 ** 30)

            get_free_percentage = int(int(used) / int(total) * 100)
            self.progress.setValue(get_free_percentage)

        except Exception as e:
            logging.warning(e)



    def recheck(self):
        MainWindow.recheck_iem = self.display_list_sq.currentItem().text()
        print(f'Rechecking {self.display_list_sq.currentItem().text()}...')
        thread_recheck = recheck_single(self)
        thread_recheck.update_item_status.connect(self.update_color)
        thread_recheck.update_f_res.connect(self.isp_res)
        thread_recheck.changePixmap.connect(self.load_display1)
        thread_recheck.change_isp_time.connect(self.set_isp_time)
        thread_recheck.save_data.connect(self.save_data)
        thread_recheck.start()

    def check_connection(self):
        try:

            port = MainWindow.io_db.all()[0]["com_scanner"]

            serialPort = serial.Serial(port, baudrate=9600,
                                       bytesize=8, timeout=0.25, stopbits=serial.STOPBITS_ONE)

            self.st_scanner.setStyleSheet("color: rgb(0, 0, 0);background-color: rgb(85, 255, 127);")
            self.st_scanner.setText(f"SCANNER OK [{port}]")
            serialPort.close()

        except Exception as e:
            self.st_scanner.setStyleSheet("color: rgb(0, 0, 0);background-color: rgb(255, 0, 0);")
            self.st_scanner.setText(f"SCANNER FAILED [{port}]")

    def rf_mapping(self):

        try:
            self.mapping_data = self.get_mapping()
        except Exception as e:
           logging.warning(e)
        # refresh mapping db



    # def create_sq(self):
    #     new_sql = self.line_new_sq.text()
    #     self.sq_db.insert({'name': new_sql, 'sequence_items': list()})
    #     self.load_cb(self.cb_sq, self.get_sq_list())
    def auto_map_sq(self):
        try:
            for i in self.mapping_data:
                if i in self.epass_input.text() and self.mapping_data[i] != self.display_current_sq.text():
                    self.write_show_log(f'AUTO CHANGE MODEL TO: {self.mapping_data[i]}')
                    index = self.cb_sq.findText(self.mapping_data[i])
                    self.cb_sq.setCurrentIndex(index)
                    self.load_sq()
                    break

        except Exception as e:
            logging.warning(e)

    def get_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = None
        finally:
            s.close()
        return IP
    #
    def open_wss(self):
        self.dialog = WSWindow()
        self.dialog.show()

    def open_dcs(self):
        self.dialog = DCWindow()
        self.dialog.show()

    def oes(self):
        self.dialog = OEWindow()
        self.dialog.show()

    def open_rv(self):
        self.dialog = PWindow()
        self.dialog.show()

    def open_ts(self):
        self.dialog = TSWindow()
        self.dialog.show()

    def open_fpb(self):
        self.dialog = FPBWindow()
        self.dialog.show()

    def get_mapping(self):
        try:
            self.init_db()
            query = f"select * from model_mapping"
            self.mycursor.execute(query)
            res = self.mycursor.fetchall()
            mapping_data = dict()
            if res is not None:
                for row in res:
                    mapping_data[row[0]] = row[1]
                return mapping_data

        except Exception as e:
            logging.warning(e)

    def get_board_mapping(self):
        try:
            self.init_db()
            query = f"select * from model_mapping "
            self.mycursor.execute(query)
            res = self.mycursor.fetchall()
            board_mapping_data = dict()
            if res is not None:
                for row in res:
                    board_mapping_data[row[0]] = row[2]
                MainWindow.board_mapping = board_mapping_data
        except Exception as e:
            logging.warning(e)


    def connect_sql(self):
        try:
            mydb = mysql.connector.connect(
                host=MainWindow.sql_db.all()[0]["host"],
                passwd=MainWindow.sql_db.all()[0]["pw"],
                port=MainWindow.sql_db.all()[0]["port"],
                database=MainWindow.sql_db.all()[0]["db"],
                user=MainWindow.sql_db.all()[0]["user"],
            )
            return mydb

        except Exception as err:
            print("Something went wrong connect_sql(): {}".format(err))

    def change_state(self):
       self.auto_change_mode = self.cb_auto_change.isChecked()
       self.write_show_log(f'AUTO CHANGE SEQUENCE: {self.auto_change_mode}')

    def convrt_cv_qt(self, img, x, y):
        try:
            rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(x, y, Qt.KeepAspectRatio)
            return p
        except Exception as e:
            logging.warning(e)

    def display_template(self, img, obj, x, y):
        try:
            image = self.convrt_cv_qt(img, x, y)
            obj.setPixmap(QPixmap.fromImage(image))
            obj.setScaledContents(True)
        except Exception as e:
            logging.warning(e)

    def show_item_info(self):
        try:
            mytree = ET.parse(resource_path(f'SEQUENCES/{self.cb_sq.currentText()}.xml'))
            root = mytree.getroot()
            for i in root[self.display_list_sq.currentRow()]:
                text += f"SPECS: {str(i.attrib['name']).upper()}: {i.text}\n"
            text += '--------------'
            self.write_show_log(text)

        except Exception as e:
            logging.warning(e)

    @pyqtSlot(str)
    def write_show_log(self, text):
        self.log_terminal.insertPlainText(text+'\n')
        self.log_terminal.moveCursor(QtGui.QTextCursor.End)

    def clear_log(self):
        try:
            self.log_terminal.clear()
        except Exception as e:
            logging.warning(e)

    def run(self):
        try:
            check_img_folder()
            self.clear_all()
            if self.auto_change_mode is True:
                self.auto_map_sq()
            if self.inspection_sq is not None:
                self.final_res = list()
                self.btn_run.setEnabled(False)

                # THREAD FOR CAMERA 1
                cam1_thread = Thread(self)
                cam1_thread.update_item_status.connect(self.update_color)
                cam1_thread.update_f_res.connect(self.isp_res)
                cam1_thread.changePixmap.connect(self.load_display1)
                cam1_thread.change_isp_time.connect(self.set_isp_time)
                cam1_thread.save_data.connect(self.save_data)
                cam1_thread.start()
                cam1_thread.finished.connect(lambda: self.btn_run.setEnabled(True))

            else:
                self.write_show_log('[Error] Please load inspection sequence!!!')
            self.update_space()
        except Exception as e:
            self.btn_run.setEnabled(True)
            logging.warning(e)

    @pyqtSlot(np.ndarray, bool, str)
    def save_data(self, image, res, TYPE):
        if TYPE != 'WAIT':
            status = {True:"OK", False:"NG"}

            try:
                if self.epass_input.text() != "":
                    save_path = resource_path(f'Images_data/{datetime.today().strftime("%m%d%Y")}/{status[res]}/{self.epass_input.text()}_{TYPE}.png')
                else:
                    save_path = resource_path(f'Images_data/{datetime.today().strftime("%m%d%Y")}/{status[res]}/{uuid.uuid4()}_{TYPE}.png')

                cv2.imwrite(save_path, image)
                logging.warning(f'[Insp Result]{self.epass_input.text()}_{TYPE}_{status[res]}')
            except Exception as e:
                logging.warning(e)


    def clear_all(self):
        self.final_res = list()
        self.display_isp_res.setText("WAITING...")
        self.display_isp_res.setStyleSheet("background-color: rgb(255, 255, 127);")
        MainWindow.result_image = dict()

    @pyqtSlot(float)
    def set_isp_time(self, time):
        try:
            self.lcdnumber.display(time*1000)
        except Exception as e:
            logging.warning(e)

    @pyqtSlot(np.ndarray)
    def load_display1(self, image):
        try:
            self.display_template(image, self.display1, 700, 700)
        except Exception as e:
            logging.warning(e)

    @pyqtSlot(bool)
    def isp_res(self, res):
        try:
            self.final_res.append(res)
            if len(self.final_res) == len(MainWindow.camera_db.all()):
                if False in self.final_res:
                    self.display_isp_res.setText("NG")
                    self.display_isp_res.setStyleSheet("background-color: rgb(255, 0, 0);")
                else:
                    self.display_isp_res.setText("OK")
                    self.display_isp_res.setStyleSheet("background-color: rgb(0, 255, 0);")

        except Exception as e:
                logging.warning(e)

    def upload_res(self, res, TYPE):
        self.init_db()
        mySql_insert_query = f"INSERT INTO isp_results (EPASS, RESULT, MACHINE_IP, ISP_MODE) VALUES ('{self.epass_input.text()}', '{res}', '{self.machine_ip}', '{TYPE}')"
        self.mycursor.execute(mySql_insert_query)
        self.mydb.commit()

    @pyqtSlot(str, bool)
    def update_color(self, text, res):
        try:
            self.upload_res(res, text)

            item = self.display_list_sq.findItems(text, QtCore.Qt.MatchExactly)[0]
            if res is False:
                item.setForeground(Qt.red)
            elif res is True:
                item.setForeground(Qt.green)
        except Exception as e:
            logging.warning(e)

    def control_sq(self):
        self.dialog = SQWindow()
        self.dialog.show()
    # def save_sq(self):
    #     self.load_insoection_sq()
    #     self.sq_db.remove(where('name') == self.cb_sq.currentText())
    #     self.sq_db.insert({'name': self.cb_sq.currentText(), 'sequence_items': self.inspection_sq})
    
    def get_sq_list(self):
        return [f.split('.')[0] for f in os.listdir(resource_path(r'SEQUENCES')) if os.path.isfile(os.path.join(resource_path(r'SEQUENCES'), f)) and '.xml' in f]

    def get_sq_items(self):
        mytree = ET.parse(resource_path(f'SEQUENCES/{self.cb_sq.currentText()}.xml'))
        return [x.tag for x in mytree.getroot()]

    def load_cb(self, obj, data):
        obj.clear()  # delete all items from comboBox
        obj.addItems(data)
        obj.setCurrentIndex(-1)

    def load_sq(self):
        try:
            data = self.get_sq_items()
            self.display_list_sq.clear()
            self.display_list_sq.addItems(data)
            self.display_current_sq.setText(self.cb_sq.currentText())
            self.load_insoection_sq()
        except Exception as e:
            logging.warning(e)

    def convrt_cv_qt(self, img, x, y):
        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(x, y, Qt.KeepAspectRatio)
        return p

    def display_template(self, img, obj, x, y):
        image = self.convrt_cv_qt(img, x, y)
        obj.setPixmap(QPixmap.fromImage(image))
        obj.setScaledContents(True)

    def load_insoection_sq(self):
        try:
            MainWindow.inspection_sq = ET.parse(resource_path(f'SEQUENCES/{self.cb_sq.currentText()}.xml')).getroot()
        except Exception as e:
            logging.warning(e)


class SQWindow(QtWidgets.QMainWindow):
    # Connect camera

    def __init__(self):
        super(SQWindow, self).__init__()
        uic.loadUi('ui/sq.ui', self)

        self.show()


class CWindow(QtWidgets.QMainWindow):
    # Connect camera

    def __init__(self):
        super(CWindow, self).__init__()
        uic.loadUi('ui/c.ui', self)

        self.show()


class TSWindow(QtWidgets.QMainWindow):
    # Connect camera

    def __init__(self):
        super(TSWindow, self).__init__()
        uic.loadUi('ui/ts.ui', self)

        self.image_label = self.findChild(QtWidgets.QLabel, 'label_2')
        self.sl_th = self.findChild(QtWidgets.QSlider, 'horizontalSlider')
        self.sl_x = self.findChild(QtWidgets.QSlider, 'horizontalSlider_2')
        self.sl_y = self.findChild(QtWidgets.QSlider, 'horizontalSlider_3')

        self.lbl_th = self.findChild(QtWidgets.QLabel, 'label_7')
        self.lbl_x = self.findChild(QtWidgets.QLabel, 'label_9')
        self.lbl_y = self.findChild(QtWidgets.QLabel, 'label_11')
        self.lbl_cnts = self.findChild(QtWidgets.QLabel, 'label_13')
        self.lbl_avg_cnts = self.findChild(QtWidgets.QLabel, 'label_15')

        self.sl_th.valueChanged.connect(self.update_th_value)
        self.sl_x.valueChanged.connect(self.update_x_value)
        self.sl_y.valueChanged.connect(self.update_y_value)


        self.timer = QtCore.QTimer(self, interval=45)
        self.timer.timeout.connect(self.update_frame)

        self.timer.start()

        self.show()

    def update_th_value(self):
        self.lbl_th.setText(str(self.sl_th.value()))
    def update_x_value(self):
        self.lbl_x.setText(str(self.sl_x.value()))
    def update_y_value(self):
        self.lbl_y.setText(str(self.sl_y.value()))

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

    def process_img(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh_frame = cv2.threshold(gray, int(self.sl_th.value()), 255, cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.erode(thresh_frame, None, iterations=1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(self.sl_x.value()), int(self.sl_y.value())))
            thresh_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_DILATE, kernel)
            cntrs = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]


            if len(cntrs) != 0:
                areas = sum([cv2.contourArea(c) for c in cntrs])/len(cntrs)
            # cv2.putText(thresh_frame, f'NUmber of contours: {len(cntrs)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, 255)
                self.lbl_cnts.setText(f'{len(cntrs)}')
                self.lbl_avg_cnts.setText(f'{str(areas)[:6]}')

            return thresh_frame

        except Exception as e:
            logging.warning(e)


    @QtCore.pyqtSlot()
    def update_frame(self):
        # image = MainWindow.viewer1.get_image()
        image = cv2.imread(r'D:\VISION INSPECTION SYSTEM\Images_data\3b5facaf-fd89-48d4-a1aa-cf8e1da54e03_CURV_LED_OFF.png')
        s = self.process_img(image)
        self.displayImage(s)

    def displayImage(self, img):
        img = cv2.resize(img, (1200, 750))
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if True:
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(outImage))


class WSWindow(QtWidgets.QMainWindow):
    # Connect camera

    def __init__(self):
        super(WSWindow, self).__init__()
        uic.loadUi('ui/tws.ui', self)

        self.image_label = self.findChild(QtWidgets.QLabel, 'label_2')
        self.sl_th = self.findChild(QtWidgets.QSlider, 'horizontalSlider')
        self.sl_x = self.findChild(QtWidgets.QSlider, 'horizontalSlider_2')
        self.sl_y = self.findChild(QtWidgets.QSlider, 'horizontalSlider_3')

        self.lbl_th = self.findChild(QtWidgets.QLabel, 'label_7')
        self.lbl_cnts = self.findChild(QtWidgets.QLabel, 'label_4')
        self.lbl_avg_cnts = self.findChild(QtWidgets.QLabel, 'label_5')

        self.sl_th.valueChanged.connect(self.update_th_value)


        self.timer = QtCore.QTimer(self, interval=45)
        self.timer.timeout.connect(self.update_frame)

        self.timer.start()

        self.show()

    def update_th_value(self):
        self.lbl_th.setText(str(self.sl_th.value()))

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

    def process_img(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, int(self.sl_th.value()), 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=2)

            cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]


            if len(cntrs) != 0:
                areas = sum([cv2.contourArea(c) for c in cntrs])/len(cntrs)
            # cv2.putText(thresh_frame, f'NUmber of contours: {len(cntrs)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, 255)
                self.lbl_cnts.setText(f'{len(cntrs)}')
                self.lbl_avg_cnts.setText(f'{str(areas)[:6]}')

            return thresh
        except Exception as e:
            logging.warning(e)


    @QtCore.pyqtSlot()
    def update_frame(self):
        image = MainWindow.viewer1.get_image()
        # image = cv2.imread(r'D:\VN39BN9652920AS474R5E0253_WHITE_SPOTS.png')
        s = self.process_img(image)
        display_image = self.displayImage(s)

    def displayImage(self, img):
        img = cv2.resize(img, (1200, 750))
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if True:
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(outImage))


class DCWindow(QtWidgets.QMainWindow):
    # Connect camera

    def __init__(self):
        super(DCWindow, self).__init__()
        uic.loadUi('ui/dc.ui', self)

        self.image_label = self.findChild(QtWidgets.QLabel, 'label_2')
        self.sl_o = self.findChild(QtWidgets.QSlider, 'horizontalSlider_3')
        self.sl_c = self.findChild(QtWidgets.QSlider, 'horizontalSlider_2')
        self.sl_V = self.findChild(QtWidgets.QSlider, 'horizontalSlider_5')
        self.sl_H = self.findChild(QtWidgets.QSlider, 'horizontalSlider_8')
        self.sl_S = self.findChild(QtWidgets.QSlider, 'horizontalSlider')

        self.lbl_o = self.findChild(QtWidgets.QLabel, 'label_14')
        self.lbl_c = self.findChild(QtWidgets.QLabel, 'label_10')
        self.lbl_V = self.findChild(QtWidgets.QLabel, 'label_21')
        self.lbl_H = self.findChild(QtWidgets.QLabel, 'label_22')
        self.lbl_S = self.findChild(QtWidgets.QLabel, 'label_23')

        self.sl_o.valueChanged.connect(self.update_o_value)
        self.sl_c.valueChanged.connect(self.update_c_value)

        self.sl_V.valueChanged.connect(self.update_V_value)
        self.sl_H.valueChanged.connect(self.update_H_value)
        self.sl_S.valueChanged.connect(self.update_S_value)


        self.timer = QtCore.QTimer(self, interval=45)
        self.timer.timeout.connect(self.update_frame)

        self.timer.start()

        self.show()

    def update_V_value(self):
        self.lbl_V.setText(str(self.sl_V.value()))

    def update_H_value(self):
        self.lbl_H.setText(str(self.sl_H.value()))

    def update_S_value(self):
        self.lbl_S.setText(str(self.sl_S.value()))

    def update_o_value(self):
        self.lbl_o.setText(str(self.sl_o.value()))
    def update_c_value(self):
        self.lbl_c.setText(str(self.sl_c.value()))

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

    def process_img(self, img):
        try:
            frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            frame_threshold = cv2.inRange(frame_HSV,( int(self.sl_H.value()),
                                                      int(self.sl_S.value()),
                                                      int(self.sl_V.value())),
                                          (255, 255, 255))


            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(self.sl_c.value()), int(self.sl_c.value())))
            closing = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel)
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (int(self.sl_o.value()), int(self.sl_o.value())))

            dil = cv2.dilate(~closing, kernel2, iterations=1)
            dil_inv = ~dil

            cntrs = cv2.findContours(dil_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            if len(cntrs) > 0:
                areas = [cv2.contourArea(c) for c in cntrs]
                max_index = np.argmax(areas)
                cnt = cntrs[max_index]
                hull = cv2.convexHull(cnt)
                cv2.drawContours(dil, [hull], 0, (0, 0, 0), 10)
            return dil
        except Exception as e:
            logging.warning(e)


    @QtCore.pyqtSlot()
    def update_frame(self):
        image = MainWindow.viewer1.get_image()
        # image = cv2.imread(r'D:\Dummy\VN39BN9653071AS474R4V0096_WHITE_SPOTS.png')
        s = self.process_img(image)
        self.displayImage(s)

    def displayImage(self, img):
        img = cv2.resize(img, (1200, 750))
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if True:
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(outImage))


class FPBWindow(QtWidgets.QMainWindow):
    # Connect camera

    def __init__(self):
        super(FPBWindow, self).__init__()
        uic.loadUi('ui/fp_board.ui', self)

        try:
            self.ip_epass = self.findChild(QtWidgets.QLineEdit, 'lineEdit')
            self.ip_epass2 = self.findChild(QtWidgets.QLineEdit, 'lineEdit_2')
            self.lbl_status = self.findChild(QtWidgets.QLabel, 'label')
            self.lbl_status_send = self.findChild(QtWidgets.QLabel, 'label_3')
            self.lbl_com_status = self.findChild(QtWidgets.QLabel, 'label_4')
            self.mycursor = None
            self.mydb = None
            self.check_connection()



        except Exception as e:
            logging.warning(e)

        # try:
        #     self.thread_epass2 = Thread_scanner2(self)
        #     self.thread_epass2.get_epass.connect(self.set_epass)
        #     self.thread_epass2.start()
        # except Exception as e:
        #     logging.warning(e)

        self.show()

    def check_connection(self):
        try:

            port = MainWindow.io_db.all()[0]["com_scanner2"]

            serialPort = serial.Serial(port, baudrate=9600,
                                       bytesize=8, timeout=0.25, stopbits=serial.STOPBITS_ONE)

            self.lbl_com_status.setStyleSheet("color: rgb(0, 0, 0);background-color: rgb(85, 255, 127);")
            self.lbl_com_status.setText("SCANNER OK")
            serialPort.close()

        except Exception as e:
            self.lbl_com_status.setStyleSheet("color: rgb(0, 0, 0);background-color: rgb(255, 0, 0);")
            self.lbl_com_status.setText("SCANNER FAILED")

    def clear(self):
        self.ip_epass2.clear()
        self.ip_epass.clear()

    @pyqtSlot(str)
    def set_epass(self, sn):
        try:
            text = re.sub("[^0-9a-zA-Z]", "", sn)
            if self.ip_epass.text() == "" and self.ip_epass2.text() == "":
                self.ip_epass.setText(text)
            elif self.ip_epass.text() != "" and self.ip_epass2.text() == "":
                self.ip_epass2.setText(text)

            if self.ip_epass.text() != "" and self.ip_epass2.text() != "":
                self.check(str(self.ip_epass.text()), str(self.ip_epass2.text()))
                self.clear()

        except Exception as e:
            logging.warning(e)

    def closeEvent(self, event):
        self.thread_epass2.quit()
        event.accept()

    def connect_sql(self):
        try:
            mydb = mysql.connector.connect(
                host=MainWindow.sql_db.all()[0]["host"],
                passwd=MainWindow.sql_db.all()[0]["pw"],
                port=MainWindow.sql_db.all()[0]["port"],
                database=MainWindow.sql_db.all()[0]["db"],
                user=MainWindow.sql_db.all()[0]["user"],
            )
            return mydb

        except Exception as err:
            print("Something went wrong connect_sql(): {}".format(err))

    def init_db(self):
        try:

            self.mydb = self.connect_sql()
            self.mycursor = self.mydb.cursor()
        except Exception as e:
            logging.warning(e)

    def check(self, a, b):
        try:
            key = a + "_" + b
            if a in MainWindow.board_mapping:
                if MainWindow.board_mapping[a] == b:
                    self.upload(key, True)
                else:
                    self.upload(key, False)
            elif b in MainWindow.board_mapping:
                if MainWindow.board_mapping[b] == a:
                    self.upload(key, True)
                else:
                    self.upload(key, False)
            else:
                self.upload(key, False)

            self.lbl_status_send.setText("UPLOAD OK")

        except Exception as e:
            self.upload(key, False)

    def upload(self, key, res):
        if res is False:
            self.lbl_status.setStyleSheet("background-color: rgb(255, 0, 0)")
            self.lbl_status.setText("NG")
        elif res is True:
            self.lbl_status.setStyleSheet("background-color: rgb(0, 255, 0)")
            self.lbl_status.setText("OK")

        self.init_db()
        mySql_insert_query = f"INSERT INTO ldboard_result (CHECK_KEY, RESULT, MACHINE_IP, DATE) VALUES ('{key}', '{res}', '{MainWindow.IP}','{datetime.now()}')"
        self.mycursor.execute(mySql_insert_query)
        self.mydb.commit()


class OEWindow(QtWidgets.QMainWindow):
    # Connect camera

    def __init__(self):
        super(OEWindow, self).__init__()
        uic.loadUi('ui/ts.ui', self)

        self.image_label = self.findChild(QtWidgets.QLabel, 'label_2')
        self.sl_th = self.findChild(QtWidgets.QSlider, 'horizontalSlider')
        self.sl_x = self.findChild(QtWidgets.QSlider, 'horizontalSlider_2')
        self.sl_y = self.findChild(QtWidgets.QSlider, 'horizontalSlider_3')

        self.lbl_th = self.findChild(QtWidgets.QLabel, 'label_7')
        self.lbl_x = self.findChild(QtWidgets.QLabel, 'label_9')
        self.lbl_y = self.findChild(QtWidgets.QLabel, 'label_11')
        self.lbl_cnts = self.findChild(QtWidgets.QLabel, 'label_13')
        self.lbl_avg_cnts = self.findChild(QtWidgets.QLabel, 'label_15')

        self.sl_th.valueChanged.connect(self.update_th_value)
        self.sl_x.valueChanged.connect(self.update_x_value)
        self.sl_y.valueChanged.connect(self.update_y_value)


        self.timer = QtCore.QTimer(self, interval=45)
        self.timer.timeout.connect(self.update_frame)

        self.timer.start()

        self.show()

    def update_th_value(self):
        self.lbl_th.setText(str(self.sl_th.value()))
    def update_x_value(self):
        self.lbl_x.setText(str(self.sl_x.value()))
    def update_y_value(self):
        self.lbl_y.setText(str(self.sl_y.value()))

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

    def process_img(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh_frame = cv2.threshold(gray, int(self.sl_th.value()), 255, cv2.THRESH_BINARY)[1]
            # thresh_frame = cv2.erode(thresh_frame, None, iterations=1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(self.sl_x.value()), int(self.sl_y.value())))
            thresh_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_DILATE, kernel)
            cntrs = cv2.findContours(thresh_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            areas = [cv2.contourArea(c) for c in cntrs]
            max_index = np.argmax(areas)
            cnt = cntrs[max_index]
            hull = cv2.convexHull(cnt)
            cv2.drawContours(thresh_frame, [hull], 0, (255, 255, 255), 10)
            # cv2.putText(thresh_frame, f'NUmber of contours: {len(cntrs)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, 255)
            self.lbl_cnts.setText(f'{len(cntrs)}')
            return thresh_frame

        except Exception as e:
            logging.warning(e)


    @QtCore.pyqtSlot()
    def update_frame(self):
        image = MainWindow.viewer1.get_image()
        #image = cv2.imread(r'D:\VISION INSPECTION SYSTEM\test\dummy27\VN39BN9652762CS803R210055_WHITE_SPOTS.png')
        s = self.process_img(image)
        self.displayImage(s)

    def displayImage(self, img):
        img = cv2.resize(img, (1200, 750))
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if True:
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(outImage))


class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)


    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)


class PWindow(QtWidgets.QWidget):
    def __init__(self):
        super(PWindow, self).__init__()
        self.resize(500, 500)
        self.viewer = PhotoViewer(self)
        # 'Load image' button
        self.setWindowTitle("IMAGE VIEWER")
        self.btnLoad = QtWidgets.QToolButton(self)
        self.cb_mode = QtWidgets.QComboBox(self)

        font = self.cb_mode.font()
        font.setPointSize(14)
        self.cb_mode.setFont(font)

        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.loadImage)
        # Button to change from drag/pan to getting pixel info
        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.cb_mode)
        VBlayout.addWidget(self.viewer)
        HBlayout = QtWidgets.QHBoxLayout()
        HBlayout.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        VBlayout.addLayout(HBlayout)
        self.load_cb()

        try:
            self.cb_mode.currentTextChanged.connect(self.loadImage)
            self.loadImage()
        except Exception as e:
            logging.warning(e)


    def load_cb(self):
        try:
            self.cb_mode.clear()
            for i in MainWindow.result_image:
                if i != 'WAIT':
                    self.cb_mode.addItem(i)

        except Exception as e:
            logging.warning(e)

    def loadImage(self):
        try:
            if self.cb_mode.currentText() != "":
                self.viewer.setPhoto(self.convert_nparray_to_QPixmap(MainWindow.result_image[self.cb_mode.currentText()]))
        except Exception as e:
            logging.warning(e)

    def convert_nparray_to_QPixmap(self, img):
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        outImage = QtGui.QPixmap.fromImage(outImage)

        return outImage


if __name__ == "__main__":
    #APP VERSION
    __version__ = '1.1.0'
    #Init LOG FOLDER
    if not os.path.exists(resource_path(r'LOG')):
        os.makedirs(resource_path(r'LOG'))

    if not os.path.exists(resource_path(r'LOG\{}.log'.format(datetime.today().strftime("%Y%m%d")))):
        log_file = open(resource_path(r'LOG\{}.log'.format(datetime.today().strftime("%Y%m%d"))), "a")

    logging.basicConfig(filename=resource_path(r'LOG\{}.log'.format(datetime.today().strftime("%Y%m%d"))),
                        filemode='a', format=f'%(asctime)s %(filename)s: %(message)s',
                        level=logging.WARNING)

    #init tf env and gpu
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    #     print(f'USING DEVICE: {gpu}')
    if not os.path.exists(resource_path(r'DATA/app_db.json')):
        db = open(resource_path('DATA/app_db.json'), 'w+')

    check_img_folder()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    if True is True:
        qtmodern.styles.dark(app)
    else:
        qtmodern.styles.light(app)

    app.exec_()
