#include "coordinatedialog.h"
#include "ui_coordinatedialog.h"

CoordinateDialog::CoordinateDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CoordinateDialog)
{
    ui->setupUi(this);
    x=0;
    y=0;
}

CoordinateDialog::~CoordinateDialog()
{
    delete ui;
}


void CoordinateDialog::setX(int x)
{
    ui->spinBox->setValue(x);
    this->x = x;
}

void CoordinateDialog::setY(int y)
{
    ui->spinBox_2->setValue(y);
    this->y = y;
}

void CoordinateDialog::execute()
{
    x = ui->spinBox->value();
    y = ui->spinBox_2->value();
}

int CoordinateDialog:: getx()
{
    return x;
}

int CoordinateDialog:: gety()
{
    return y;
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
