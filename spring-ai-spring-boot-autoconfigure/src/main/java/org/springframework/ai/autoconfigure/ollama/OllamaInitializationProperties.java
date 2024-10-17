/*
 * Copyright 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.autoconfigure.ollama;

import org.springframework.ai.ollama.management.PullModelStrategy;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;

/**
 * Ollama initialization configuration properties.
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
@ConfigurationProperties(OllamaInitializationProperties.CONFIG_PREFIX)
public class OllamaInitializationProperties {

	public static final String CONFIG_PREFIX = "spring.ai.ollama.init";

	/**
	 * Whether to pull models at startup-time and how.
	 */
	private PullModelStrategy pullModelStrategy = PullModelStrategy.NEVER;

	/**
	 * How long to wait for a model to be pulled.
	 */
	private Duration timeout = Duration.ofMinutes(5);

	/**
	 * Maximum number of retries for the model pull operation.
	 */
	private int maxRetries = 0;

	public PullModelStrategy getPullModelStrategy() {
		return pullModelStrategy;
	}

	public void setPullModelStrategy(PullModelStrategy pullModelStrategy) {
		this.pullModelStrategy = pullModelStrategy;
	}

	public Duration getTimeout() {
		return timeout;
	}

	public void setTimeout(Duration timeout) {
		this.timeout = timeout;
	}

	public int getMaxRetries() {
		return maxRetries;
	}

	public void setMaxRetries(int maxRetries) {
		this.maxRetries = maxRetries;
	}

}
