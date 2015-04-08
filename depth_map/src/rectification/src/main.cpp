#include <iostream>

#include "image_pair.hpp"

inline void EnableMemLeakCheck(void)
{
	_CrtSetDbgFlag(_CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) | _CRTDBG_LEAK_CHECK_DF);
	//_CrtSetBreakAlloc(3267);
}

void usage()
{
	std::cout<<"usage:"<<std::endl;
	std::cout<<"\t executable_name yaml_name"<<std::endl;
}

int main(int _argc, char ** _argv)
{
	EnableMemLeakCheck();

	if (_argc != 2)
	{
		usage();
		return EXIT_FAILURE;
	}

	
	std::string yaml_filename = _argv[1];
	rect::ImagePair pair(yaml_filename);
	//pair.SaveRectifiedPair("test");
	if (pair.SimpleDaisyDense())
	{
		std::cout<<"failed."<<std::endl;
	}

	std::cout<<"rectification."<<std::endl;
	return EXIT_SUCCESS;
}