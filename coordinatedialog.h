#ifndef COORDINATEDIALOG_H
#define COORDINATEDIALOG_H

#include <QDialog>
#include <QString>

namespace Ui {
class CoordinateDialog;
}

class CoordinateDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CoordinateDialog(QWidget *parent = 0);

    ~CoordinateDialog();
    void setX(int x);
    void setY(int y);
    void execute();
    void getx(float* there);
    void gety(float* there);
    void setXDestination(float *xPointer);
    void setYDestination(float* yPointer);
    void setLabel( QString string );

private slots:
    void on_buttonBox_accepted();

private:
    Ui::CoordinateDialog *ui;
    float x,y;
    float* putXHere;
    float* putYHere;
};

#endif // COORDINATEDIALOG_H
