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
    void getx(int* there);
    void gety(int* there);
    void setXDestination(int* xPointer);
    void setYDestination(int* yPointer);
    void setLabel( QString string );

private slots:
    void on_buttonBox_accepted();

private:
    Ui::CoordinateDialog *ui;
    int x,y;
    int* putXHere;
    int* putYHere;
};

#endif // COORDINATEDIALOG_H
