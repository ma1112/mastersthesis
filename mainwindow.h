#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "image.h"
#include <QLabel>
#include <vector>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();





private slots:
    void on_button_choosefile_clicked();

    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;
    Image image;
    std::vector<Image> images_temp, images;

};

#endif // MAINWINDOW_H
