package webapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import springfox.documentation.swagger2.annotations.EnableSwagger2;

/**
 * VnCoreNLP REST API Application
 * Spring Boot application providing REST endpoints for Vietnamese NLP processing.
 */
@SpringBootApplication
@EnableSwagger2
public class VnCoreNLPWebAPI {
    
    public static void main(String[] args) {
        SpringApplication.run(VnCoreNLPWebAPI.class, args);
    }
}
