#pragma once

#include"Functor.h"
#include"MainWindow.h"
#include<QApplication>
#include<gl/GL.h>
#include<gl/GLU.h>

int main(int argc, char* argv[])
{
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_DEPTH | GLUT_RGBA | GLUT_DOUBLE);
    getDeviceCount();
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QApplication app(argc, argv);

    new MainWindow();

    return app.exec();
}