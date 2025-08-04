#include "Object.h"

//float Object::m_screenRatio = 1.0f;
//QMatrix4x4 Object::m_model = QMatrix4x4();
Object::Stopwatch<> Object::stopwatch;

Object::Object(): m_vertexCount(0), m_indexCount(0)//, m_colorChannel(0)
{
	//m_pVertices = nullptr;
	m_Context = nullptr;
	m_pIndices = nullptr;
	m_program = nullptr;
	//m_Zoom = nullptr;
	
	//m_scale = nullptr;
	//m_centre = nullptr;

	//m_model.setToIdentity();
	//m_model.ortho(-0.5f, +0.5f, +0.5f, -0.5f, -1.0f, 1.0f);
	//
	//m_scale.setX(0.1);
	//m_scale.setY(0.1);
	//m_scale.setZ(0.1);
	//m_pFunctions = nullptr;
}
Object::Object(QOpenGLContext* context) : m_Context(context), m_vertexCount(0), m_indexCount(0)//, m_colorChannel(0)
{
	//m_pVertices = nullptr;
	//m_Context->setShareContext(context);
	m_pIndices = nullptr;
	m_program = nullptr;
	//m_Zoom = nullptr;
	//m_scale = nullptr;
	//m_centre = nullptr;
	
	
	//m_model.ortho(-0.5f, +0.5f, +0.5f, -0.5f, -1.0f, 1.0f);
	//
	//m_scale.setX(0.1);
	//m_scale.setY(0.1);
	//m_scale.setZ(0.1);
	//m_pFunctions = nullptr;
}

//Object::Object(float* position, float* color, float* normal, const int& vertexCount, const int& colorChannel)
//{
//	if (vertexCount > 0)
//	{
//		m_vertexCount = vertexCount;
//		m_pVertices = new VertexType[m_vertexCount * 3];
//		std::copy(position, position + m_vertexCount * 3, m_pVertices->position);
//
//
//		
//	}
//}

Object::Object(const Object& other)
{
	//this->ReleaseBuffers();
	this->copy(other);
}

Object::~Object()
{
	//ShutDown();
	//m_colorChannel = 0;

	//SafeReleasePointer(m_model);
	//SafeReleasePointer(m_scale);
	//SafeReleasePointer(m_centre);
	//SafeReleasePointer(m_Zoom);

	m_indexCount = 0;
	m_vertexCount = 0;
}

//void Object::allocate()
//{
//	
//}

void Object::setVertexCount(unsigned val)
{
	this->m_vertexCount = val;
}

void Object::setIndexCount(const Idx& val)
{
	this->m_indexCount = val;
}

void Object::Bind()
{
	//m_vao.bind();

	QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);
	m_vbo.bind();
	if(m_program)
		m_program->bind();
}

//void Object::Release()
//{
//
//}

void Object::ShutDown()
{
	Release();
	ReleaseGLFunctions();
	ReleaseBuffers();
}

//void Object::allocateVertices(void* other)
//{
//	std::copy((VertexType*)other, (VertexType*)other + this->m_vertexCount, (VertexType*)this->m_pVertices);
//}

void Object::initializeShader(QOpenGLShader::ShaderTypeBit type, const char* src)
{
	if (!m_program)
	{
		m_program = new QOpenGLShaderProgram(this);

	}
	m_program->addShaderFromSourceCode(type, src);
}

void Object::allocateIndices(Idx* other)
{
	std::copy(other, other+ this->m_indexCount, this->m_pIndices);
}

void Object::copy(const Object& other)
{
	this->m_vertexCount = other.getVertexCount();
	this->m_indexCount = other.getIndexCount();
	//this->m_colorChannel= other.getColorChannel();


	//this->m_pVertices = new VertexType[this->m_vertexCount];
	this->m_pIndices = new Idx[this->m_indexCount];
	
	//this->allocateVertices(other.m_pVertices);
	this->allocateIndices(other.m_pIndices);
	

}

//void Object::setColorChannel(unsigned val)
//{
//	this->m_colorChannel = val;
//}

//void Object::ReleaseGLFunctions()
//{
//	m_vbo.destroy();
//	m_vao.destroy();
//	SafeReleasePointer(m_program);
//	
//}

//void Object::ReleaseBuffers()
//{
//	SafeReleaseArray(m_pIndices);
//	SafeReleaseArray(m_pRawIndices);
//}

void Object::setContext(QOpenGLContext* context)
{
	m_Context = context;
}

const int Object::getVertexCount() const
{
	return this->m_vertexCount;
}

const int Object::getIndexCount() const
{
	return this->m_indexCount;
}

const int Object::getWidth() const
{
	return this->m_lfWidth;
}

const int Object::getHeight() const
{
	return this->m_lfHeight;
}

//const void Object::setModelMatrix(const QMatrix4x4& mat)
//{
//	//std::copy(&mat, &mat + sizeof(QMatrix4x4), *this->m_model);
//	//Object::m_model = mat;
//}
//
//const void Object::setScreenRatio(const float& _val)
//{
//	//Object::m_screenRatio = _val;
//}
//
//float Object::getScreenRatio()
//{
//	//return Object::m_screenRatio;
//}
//
//QMatrix4x4 Object::getModelMatrix()
//{
//	// TODO: insert return statement here
//	
//	return Object::m_model;
//}

//const int Object::getColorChannel() const
//{
//	return this->m_colorChannel;
//}

//unsigned long* Object::getIndices()
//{
//	return this->m_pIndices;
//}
//
//VertexType* Object::getConstData()
//{
//	return this->m_pVertices;
//}
