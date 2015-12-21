#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "image.h"
#include <QLabel>

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

private:
    Ui::MainWindow *ui;
    Image image;

};

#endif // MAINWINDOW_H
