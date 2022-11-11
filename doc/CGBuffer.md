##TODO List

```C++

typedef float real;

AxVector2UI ThreadBlockInfo(AxUInt32 blockSize, uInt64 numThreads)
{
	return MakeVector2UI(int(numThreads / blockSize) + 1, blockSize > numThreads ? numThreads : blockSize);
}

class CGBufferBase
{
public:
    void SetName();
    std::string GetName();
    virtual uint32 GetBufferSize();
    virtual bool LoadToDevice()		  { return false; };
	virtual bool LoadToHost()		  { return false; };

    static CGBufferBase* Load(std::string path);
}


template<class T>
class CGBuffer :public CGBufferBase
{
    static CGBuffer<T>* Load(std::string path);
    std::vector<T> m_Data;
    virtual bool LoadToDevice()		  { return false; };
	virtual bool LoadToHost()		  { return false; };
};

class Geometry
{
    //Buffer : vector3 , float ,string
    //
    foreach()
    {
        buffer->Read();
        buffer->save();
        buffer->LoadToDevice();
    }
}

typedef CGBuffer<glm::vec3> BufferV3;
typedef CGBuffer<real> BufferReal;
typedef CGBuffer<float> BufferFp32;
typedef CGBuffer<double> BufferFp64;

```
```C++
int main()
{
    //load from disk
    auto posBuffer = (BufferV3*)CGBufferBase::Load("D:/CodeGenerator/P.rbuf");
    auto massBuffer = (BufferFp32*)CGBufferBase::Load("D:/CodeGenerator/Mass.rbuf");

    BufferV3::Load("D:/CodeGenerator/P.rbuf");
    BufferFp32::Load("D:/CodeGenerator/Mass.rbuf");

    posBuffer.LoadToDevice();
    posBuffer.LoadToHost();
    posBuffer.SaveToOBJ("D:/CodeGenerator/p.obj");
}

```

