#include "coordinatedialog.h"
#include "ui_coordinatedialog.h"
#include "iostream"

CoordinateDialog::CoordinateDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CoordinateDialog)
{
    ui->setupUi(this);
    x=0;
    y=0;
    putXHere = NULL;
    putYHere = NULL;
}



CoordinateDialog::~CoordinateDialog()
{
    delete ui;
}


void CoordinateDialog::setX(int x)
{
    ui->spinBox_2->setValue(x);
    this->x = x;
}

void CoordinateDialog::setY(int y)
{
    ui->spinBox->setValue(y);
    this->y = y;
}

void CoordinateDialog::execute()
{
    if( putXHere != NULL && putYHere != NULL)
    {
        x = ui->spinBox_2->value();
        y = ui->spinBox->value();
        getx(putXHere);
        gety(putYHere);
        close();
    }
    else
        std::cout << "ERROR! Initlaize putXHere and putYHere first. ";

}

void CoordinateDialog:: getx( int* there )
{
    *there = x;
}

void CoordinateDialog:: gety(int* there)
{
    *there = y;
}

void CoordinateDialog::setLabel( QString string )
{
    ui->label_3->setText(string);
}



void CoordinateDialog::on_buttonBox_accepted()
{
    execute();
    ui->spinBox->setDisabled(true);
    ui->spinBox_2->setDisabled(true);
}

void CoordinateDialog::setXDestination(int* xPointer) {putXHere = xPointer; }
void CoordinateDialog::setYDestination(int* yPointer){putYHere = yPointer; }
