from functools import partial
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QImageReader
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PyQt5 import QtGui
from model.HashableQListWidgetItem import HashableQListWidgetItem
from model.LabelFile import LabelFile, LabelFileFormat, LabelFileError
from model.LableDialog import LabelDialog
from model.Shape import Shape
from utils.create_ml_io import JSON_EXT, CreateMLReader
from utils.pascal_voc_io import XML_EXT, PascalVocReader
from utils.yolo_io import TXT_EXT, YoloReader
from UI.Annotation.AnnotationWin import AnnotationWin
from utils.CommonHelper import CommonHelper
import os
import cv2
from utils.ustr import ustr
from utils.utils import generate_color_by_text, natural_sort

FORMAT_PASCALVOC='PascalVOC'
FORMAT_YOLO='YOLO'
FORMAT_CREATEML='CreateML'
__appname__ ="SMMD"
class Annotation(AnnotationWin):
    FIT_WINDOW, MANUAL_ZOOM = list(range(2))
    def __init__(self,configuration=None,default_filename =None):
        super(Annotation, self).__init__()
        self.setupUi(self)
        styleFile = '../../resource/Annotation.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.setStyleSheet(style)
        self.zoom_widget.valueChanged.connect(self.paint_canvas)
        self.zoom_in.clicked.connect(lambda :self.addZoom(10))
        self.zoom_out.clicked.connect(lambda :self.addZoom(-10))
        self.fit_window.clicked.connect(lambda :self.set_fit_window(True))
        self.createBox.clicked.connect(self.create_shape)

        self.scalers = {
            self.FIT_WINDOW: self.scale_fit_window,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }
        self._beginner = True
        self.configuration = configuration
        self.open.clicked.connect(self.open_file)
        self.openDir.clicked.connect(self.open_dir_dialog)
        self.previous.clicked.connect(self.open_prev_image)
        self.next.clicked.connect(self.open_next_image)
        self.save.clicked.connect(self.save_file)
        self.filename = None
        self.cwd = os.getcwd()  # 获取当前程序文件位置
        self.imageDir =''
        self.imageList = []
        self.imageTotal = 0
        self.imageDirPathBuffer = ''
        self.line_color = QColor(255,0,0,100)
        self.fill_color = QColor(0,255,0,100)
        self.label_hist = []
        self.canvas.newShape.connect(self.new_shape)
        self.prev_label_text = ""
        self.items_to_shapes = {}
        self.shapes_to_items = {}
        self._no_selection_slot = False
        # Whether we need to save or not.
        self.dirty = False
        self.label_file_format =LabelFileFormat.PASCAL_VOC
        self.default_save_dir = str(self.configuration.value('SAVE_DATASET_PATH'))+'/Annotations/'
        self.file_path = ustr(default_filename)
        self.deleteBox.clicked.connect(self.delete_selected_shape)
        self.canvas.drawingPolygon.connect(self.toggle_drawing_sensitive)
        self.file_list_widget.itemDoubleClicked.connect(self.file_item_double_clicked)
        if self.file_path and os.path.isdir(self.file_path):
            self.queue_event(partial(self.import_dir_images, self.file_path or ""))
        elif self.file_path:
            self.queue_event(partial(self.load_image, self.file_path or ""))

        self.info.itemActivated.connect(self.label_selection_changed)
        self.info.itemSelectionChanged.connect(self.label_selection_changed)
        self.info.itemDoubleClicked.connect(self.edit_label)
        # # Connect to itemChanged to detect checkbox changes.
        self.info.itemChanged.connect(self.label_item_changed)
        self.previous.setEnabled(False)
        self.next.setEnabled(False)
        self.save.setEnabled(False)
        self.fit_window.setEnabled(False)
        self.zoom_in.setEnabled(False)
        self.zoom_out.setEnabled(False)
    def open_file(self, _value=False):
        if not self.may_continue():
            return
        path = os.path.dirname(ustr(self.file_path)) if self.file_path else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.cur_img_idx = 0
            self.imageTotal = 1
            self.load_image(filename)

    def load_image(self,file_path):
        self.canvas.setEnabled(False)
        self.reset_state()
        file_path = ustr(file_path)
        unicode_file_path = ustr(file_path)
        unicode_file_path = os.path.abspath(unicode_file_path)
        if unicode_file_path and self.file_list_widget.count() > 0:
            if unicode_file_path in self.m_img_list:
                index = self.m_img_list.index(unicode_file_path)
                file_widget_item = self.file_list_widget.item(index)
                file_widget_item.setSelected(True)
            else:
                self.file_list_widget.clear()
                self.m_img_list.clear()

        if unicode_file_path and os.path.exists(unicode_file_path):
            if LabelFile.is_label_file(unicode_file_path):
                try:
                    self.label_file = LabelFile(unicode_file_path)
                except LabelFileError as e:
                    self.error_message(u'Error opening file',
                                       (u"<p><b>%s</b></p>"
                                        u"<p>Make sure <i>%s</i> is a valid label file.")
                                       % (e, unicode_file_path))
                    return False
                self.image_data = self.label_file.image_data
                self.line_color = QColor(*self.label_file.lineColor)
                self.fill_color = QColor(*self.label_file.fillColor)
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.img = cv2.imread(file_path)
                self.image_data = self.img.data
                self.image = QtGui.QImage( self.img.data,  self.img.shape[1],  self.img.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.file_path = unicode_file_path
            self.canvas.load_pixmap(QtGui.QPixmap.fromImage(self.image))
            if self.label_file:
                self.load_labels(self.label_file.shapes)
            self.dirty = False
            self.canvas.setEnabled(True)
            self.adjust_scale(initial=True)
            self.paint_canvas()
            self.show_label.setText(str(self.cur_img_idx+1) + "/" + str(self.imageTotal))
            self.info.clear()
            self.show_bounding_box_from_annotation_file(file_path)

    def reset_state(self):
        self.items_to_shapes.clear()
        self.shapes_to_items.clear()
        self.info.clear()
        self.file_path = None
        self.label_file = None
        self.canvas.reset_state()


    def add_label(self,shape):
        item = HashableQListWidgetItem(shape.label)
        self.items_to_shapes[item] = shape
        self.shapes_to_items[shape] = item
        self.info.addItem(item)

    def save_file(self, _value=False):
        self.default_save_dir = self.configuration.value('SAVE_DATASET_PATH') + '/Annotations/'
        if self.default_save_dir is not None and len(ustr(self.default_save_dir)):
            if not os.path.exists(self.default_save_dir):
                os.makedirs(self.default_save_dir)
            if self.file_path:
                image_file_name = os.path.basename(self.file_path)
                saved_file_name = os.path.splitext(image_file_name)[0]
                saved_path = os.path.join(ustr(self.default_save_dir), saved_file_name)
                self._save_file(saved_path)
        else:
            image_file_dir = os.path.dirname(self.file_path)
            image_file_name = os.path.basename(self.file_path)
            saved_file_name = os.path.splitext(image_file_name)[0]
            saved_path = os.path.join(image_file_dir, saved_file_name)
            self._save_file(saved_path if self.label_file
                            else self.save_file_dialog(remove_ext=False))

    def save_file_dialog(self, remove_ext=True):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        open_dialog_path = self.current_path()
        dlg = QFileDialog(self, caption, open_dialog_path, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filename_without_extension = os.path.splitext(self.file_path)[0]
        dlg.selectFile(filename_without_extension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            full_file_path = ustr(dlg.selectedFiles()[0])
            if remove_ext:
                return os.path.splitext(full_file_path)[0]  # Return file path without the extension.
            else:
                return full_file_path
        return ''

    def current_path(self):
        return os.path.dirname(self.file_path) if self.file_path else '.'


    def _save_file(self, annotation_file_path):
        if annotation_file_path and self.save_labels(annotation_file_path):
            self.dirty = False


    def save_labels(self, annotation_file_path):
        annotation_file_path = ustr(annotation_file_path)
        if self.label_file is None:
            self.label_file = LabelFile()
            self.label_file.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                        # add chris
                        difficult=s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add different annotation formats here
        try:
            if self.label_file_format == LabelFileFormat.PASCAL_VOC:
                if annotation_file_path[-4:].lower() != ".xml":
                    annotation_file_path += XML_EXT
                self.label_file.save_pascal_voc_format(annotation_file_path, shapes, self.file_path, self.image_data,
                                                       self.line_color.getRgb(), self.fill_color.getRgb())
            elif self.label_file_format == LabelFileFormat.YOLO:
                if annotation_file_path[-4:].lower() != ".txt":
                    annotation_file_path += TXT_EXT
                self.label_file.save_yolo_format(annotation_file_path, shapes, self.file_path, self.image_data,
                                                 self.label_hist,
                                                 self.line_color.getRgb(), self.fill_color.getRgb())
            elif self.label_file_format == LabelFileFormat.CREATE_ML:
                if annotation_file_path[-5:].lower() != ".json":
                    annotation_file_path += JSON_EXT
                self.label_file.save_create_ml_format(annotation_file_path, shapes, self.file_path, self.image_data,
                                                      self.label_hist, self.line_color.getRgb(),
                                                      self.fill_color.getRgb())
            else:
                self.label_file.save(annotation_file_path, shapes, self.file_path, self.image_data,
                                     self.line_color.getRgb(), self.fill_color.getRgb())
            print('Image:{0} -> Annotation:{1}'.format(self.file_path, annotation_file_path))
            return True
        except LabelFileError as e:
            self.error_message(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def paint_canvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoom_widget.value()
        self.canvas.label_font_size = int(0.02 * max(self.image.width(), self.image.height()))
        self.canvas.adjustSize()
        self.canvas.update()

    def addZoom(self,increment=10):
        self.zoom_mode = self.MANUAL_ZOOM
        self.zoom_widget.setValue(self.zoom_widget.value() + increment)

    def set_fit_window(self, value=True):
        # if value:
        #     self.actions.fitWidth.setChecked(False)
        self.zoom_mode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjust_scale()

    def adjust_scale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoom_mode]()
        # value = self.scale_fit_window()
        self.zoom_widget.setValue(int(100 * value))

    def scale_fit_window(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.width()-self.info.width() - e
        h1 = self.height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def create_shape(self):
        assert self.beginner()
        self.canvas.set_editing(False)
        self.save.setEnabled(True)


    def beginner(self):
        return self._beginner

        # Callback functions:

    def new_shape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """

        self.label_dialog = LabelDialog(
                    parent=self, list_item=self.label_hist)
        text = self.label_dialog.pop_up(text=self.prev_label_text)
        if text is not None:
            generate_color = generate_color_by_text(text)
            shape = self.canvas.set_last_label(text, generate_color,generate_color)
            self.add_label(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.set_editing(True)
            self.set_dirty()
            if text not in self.label_hist:
                self.label_hist.append(text)
        else:
            self.canvas.reset_all_lines()

    def set_dirty(self):
        self.dirty = True

    def delete_selected_shape(self):
        self.remove_label(self.canvas.delete_selected())
        self.canvas.delete_selected()
        self.set_dirty()

    def remove_label(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapes_to_items[shape]
        self.info.takeItem(self.info.row(item))
        del self.shapes_to_items[shape]
        del self.items_to_shapes[item]


    def reset_all_lines(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.set_open()
        self.line.points = [self.current[-1], self.current[0]]
        self.drawingPolygon.emit(True)
        self.current = None
        self.drawingPolygon.emit(False)
        self.update()

    def toggle_drawing_sensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""

        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.set_editing(True)
            self.canvas.restore_cursor()


    def open_dir_dialog(self, _value=False, dir_path=None, silent=False):
        default_open_dir_path = os.path.dirname(self.file_path) if self.file_path else '.'
        if silent != True:
            target_dir_path = ustr(QFileDialog.getExistingDirectory(self,
                                                                    '%s - Open Directory' % __appname__, default_open_dir_path,
                                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            target_dir_path = ustr(default_open_dir_path)
        self.last_open_dir = target_dir_path
        if target_dir_path:
            self.import_dir_images(target_dir_path)
            self.previous.setEnabled(True)
            self.next.setEnabled(True)
            self.zoom_in.setEnabled(True)
            self.zoom_out.setEnabled(True)
            self.fit_window.setEnabled(True)

    def import_dir_images(self, dir_path):

        self.last_open_dir = dir_path
        self.dir_name = dir_path
        self.file_path = None
        self.file_list_widget.clear()
        self.m_img_list = self.scan_all_images(dir_path)
        self.imageTotal = len(self.m_img_list)
        self.open_next_image()
        for imgPath in self.m_img_list:
            item = QListWidgetItem(imgPath)
            self.file_list_widget.addItem(item)


    def scan_all_images(self, folder_path):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relative_path = os.path.join(root, file)
                    path = ustr(os.path.abspath(relative_path))
                    images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def open_next_image(self, _value=False):
        if not self.may_continue():
            return
        filename = None
        if self.file_path is None:
            filename = self.m_img_list[0]
            self.cur_img_idx = 0
        else:
            if self.cur_img_idx + 1 < self.imageTotal:
                self.cur_img_idx += 1
                filename = self.m_img_list[self.cur_img_idx]

        if filename:
            self.load_image(filename)

    def open_prev_image(self, _value=False):
        # Proceeding prev image without dialog if having any label

        if self.imageTotal <= 0:
            return

        if self.file_path is None:
            return

        if self.cur_img_idx - 1 >= 0:
            self.cur_img_idx -= 1
            filename = self.m_img_list[self.cur_img_idx]
            if filename:
                self.show_label
                self.load_image(filename)


    def file_item_double_clicked(self, item=None):
        self.cur_img_idx = self.m_img_list.index(ustr(item.text()))
        filename = self.m_img_list[self.cur_img_idx]
        if filename:
            self.load_image(filename)

    def edit_label(self):
        if not self.canvas.editing():
            return
        item = self.current_item()
        if not item:
            return
        text = self.label_dialog.pop_up(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generate_color_by_text(text))
            self.set_dirty()

    def label_item_changed(self, item):
        shape = self.items_to_shapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generate_color_by_text(shape.label)
            self.set_dirty()


    def label_selection_changed(self):
        item = self.current_item()
        if item and self.canvas.editing():
            self._no_selection_slot = True
            self.canvas.select_shape(self.items_to_shapes[item])
            shape = self.items_to_shapes[item]

    def current_item(self):
        items = self.info.selectedItems()
        if items:
            return items[0]
        return None

        # React to canvas signals.

    def shape_selection_changed(self, selected=False):
        if self._no_selection_slot:
            self._no_selection_slot = False
        else:
            shape = self.canvas.selected_shape
            if shape:
                self.shapes_to_items[shape].setSelected(True)
            else:
                self.info.clearSelection()


    def may_continue(self):
        if not self.dirty:
            return True
        else:
            discard_changes = self.discard_changes_dialog()
            if discard_changes == QMessageBox.No:
                return True
            elif discard_changes == QMessageBox.Yes:
                self.save_file()
                return True
            else:
                return False

    def discard_changes_dialog(self):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        msg = u'You have unsaved changes, would you like to save them and proceed?'
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel)

    def show_bounding_box_from_annotation_file(self, file_path):
        if self.default_save_dir is not None:
            basename = os.path.basename(os.path.splitext(file_path)[0])
            xml_path = os.path.join(self.default_save_dir, basename + XML_EXT)
            txt_path = os.path.join(self.default_save_dir, basename + TXT_EXT)
            json_path = os.path.join(self.default_save_dir, basename + JSON_EXT)

            """Annotation file priority:
            PascalXML > YOLO
            """
            if os.path.isfile(xml_path):
                self.load_pascal_xml_by_filename(xml_path)
            elif os.path.isfile(txt_path):
                self.load_yolo_txt_by_filename(txt_path)
            elif os.path.isfile(json_path):
                self.load_create_ml_json_by_filename(json_path, file_path)

        else:
            xml_path = os.path.splitext(file_path)[0] + XML_EXT
            txt_path = os.path.splitext(file_path)[0] + TXT_EXT
            if os.path.isfile(xml_path):
                self.load_pascal_xml_by_filename(xml_path)
            elif os.path.isfile(txt_path):
                self.load_yolo_txt_by_filename(txt_path)

    def load_pascal_xml_by_filename(self, xml_path):
        if self.file_path is None:
            return
        if os.path.isfile(xml_path) is False:
            return

        t_voc_parse_reader = PascalVocReader(xml_path)
        shapes = t_voc_parse_reader.get_shapes()
        self.load_labels(shapes)
        self.canvas.verified = t_voc_parse_reader.verified

    def load_yolo_txt_by_filename(self, txt_path):
        if self.file_path is None:
            return
        if os.path.isfile(txt_path) is False:
            return


        t_yolo_parse_reader = YoloReader(txt_path, self.image)
        shapes = t_yolo_parse_reader.get_shapes()
        print(shapes)
        self.load_labels(shapes)
        self.canvas.verified = t_yolo_parse_reader.verified

    def load_create_ml_json_by_filename(self, json_path, file_path):
        if self.file_path is None:
            return
        if os.path.isfile(json_path) is False:
            return

        # self.set_format(FORMAT_CREATEML)

        create_ml_parse_reader = CreateMLReader(json_path, file_path)
        shapes = create_ml_parse_reader.get_shapes()
        self.load_labels(shapes)
        self.canvas.verified = create_ml_parse_reader.verified

    def load_labels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snap_point_to_canvas(x, y)
                if snapped:
                    self.set_dirty()

                shape.add_point(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generate_color_by_text(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generate_color_by_text(label)

            self.add_label(shape)
        self.canvas.load_shapes(s)